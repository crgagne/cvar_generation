from pathlib import Path
import argparse
from helpers import set_seeds
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

import os
from scipy.special import softmax
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import pickle
import pandas as pd

# from torch.utils.data.dataloader import default_collate this doesn't work with hugging face dataset
from transformers import default_data_collator
from batch_datasets import get_batch_dataset
from rl_learner import TD_Learner

EMOTIONS = ['admiration','amusement','anger','annoyance','approval','caring',
 'confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment',
 'excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride',
 'realization','relief','remorse','sadness','surprise','neutral']

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
    # consider nn.functional.HuberLoss(diff) # reduction = None

def prepare_data(args, tokenizer, split='train', subset=False):

    if args.mdp_mode:

        states = np.load('mscl/simple_mdp_mix2_states_onehot.npy')
        rewards = np.load('mscl/simple_mdp_mix2_rewards.npy')
        mask = np.load('mscl/simple_mdp_mix2_mask.npy')
        n_states = state_dim = states.shape[-1]
        dataset = TensorDataset(torch.Tensor(states),
                                torch.Tensor(rewards),
                                torch.Tensor(mask))
        def collate_with_extra_empty(batch, str_column = 'text'):
            return torch.utils.data.default_collate(batch), None
        train_data = DataLoader(dataset, collate_fn=collate_with_extra_empty, batch_size=args.batch_size, shuffle=True)

    else:

        # load dataset
        dataset = get_batch_dataset(args.data, split=split)

        if subset:
            dataset.select(range(50_000))

        # tokenize
        def tokenize(batch):
           return tokenizer(batch['text'], truncation=True, max_length=args.max_length, padding='max_length')
        dataset = dataset.map(tokenize, num_proc=1, batched=True)

        if args.filter != None:
            dataset = dataset.filter(lambda example: example['text'].startswith(args.filter))

        # tensorize sentiment
        def tensorize(batch):
            batch['sentiment']=[torch.Tensor([sentiment]) for sentiment in batch['sentiment']]
            if EMOTIONS[0] in batch:
                for emotion in EMOTIONS:
                    batch[emotion]=[torch.Tensor([sentiment]) for sentiment in batch[emotion]]
            return(batch)
        dataset = dataset.map(tensorize, num_proc=1, batched=True)

        # filter potentially #
        if args.filter_out is not None:
            dataset = dataset.filter(lambda example: not example['text'].startswith(args.filter_out))

        if args.more_balanced=='True':

            print(f' datatset size {len(dataset)}')
            def tmp_filter_func(example):
                keep=True
                p = 0.5 / len(args.emotion_set) # so it's similar for single / double emotions
                for emotion in args.emotion_set:
                    if example[emotion][0]<0.05:
                        if np.random.binomial(p=p,n=1)==1:
                            keep=False
                    if example[emotion][0]>0.05: # otherwise it will remove some of the other category
                        keep=True

                return(keep)
            dataset = dataset.filter(tmp_filter_func)
            print(f' datatset size {len(dataset)}')

        # batch data
        def collate_with_strings(batch, str_column = 'text'):
            new_batch = []; strings = []
            for _batch in batch:
                strings.append(_batch[str_column])
                _batch.pop(str_column, None)
                new_batch.append(_batch)
            return default_data_collator(new_batch), strings
        train_data = DataLoader(dataset, collate_fn=collate_with_strings, batch_size=args.batch_size, shuffle=True)
        state_dim = None

        if split=='validation':
            return(train_data, state_dim, dataset)
        else:
            return(train_data, state_dim)

def calc_state_from_batch(batch, device, model, mdp_mode=False, emotion='None'):

    if mdp_mode:
        states = batch[0].to(device)
        rewards = batch[1].to(device)
        mask = batch[2].unsqueeze(-1).to(device)

    else:
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        if emotion=='None':
            rewards = batch['sentiment'].to(device)
        elif '+' in emotion:
            emotion_set = emotion.split('+')
            rewards = torch.max(batch[emotion_set[0]],batch[emotion_set[1]])
        else:
            rewards = batch[emotion].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=mask,
                           output_hidden_states=True)

        # feed into TD learenr
        states = output['hidden_states'][-1]
        mask = mask.unsqueeze(-1)
        assert len(states.shape)==len(mask.shape)

    return(states, mask, rewards, input_ids)

def append_to_log(log_dict, key, value):
    if key not in log_dict.keys():
        log_dict[key]=[value]
    else:
        log_dict[key].append(value)
    return(log_dict)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2-large")
    parser.add_argument("--emotion", type=str, default="None")
    #parser.add_argument("--data", type=str, default='data/results/single_sentences_IYou_6/full_generations.txt')
    parser.add_argument("--data", type=str, default='/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_v1_w_emosent_subsample')
    parser.add_argument("--results_folder", type=str, default='data/results/single_sentences_posts_v1')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_quantiles", type=int, default=20)
    parser.add_argument("--mdp_mode", action='store_true')
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--target_every", type=int, default=10)
    parser.add_argument("--huber_k", type=float, default=0.1)
    parser.add_argument("--filter_out", type=str, default=None)
    parser.add_argument("--more_balanced", type=str, default=None)
    parser.add_argument("--subset", action='store_true')

    args = parser.parse_args()
    extra_save = '_linear' if args.linear else '_'+str(args.hidden_dim)
    extra_save += '_'+args.filter.replace(' ','_') if args.filter is not None else ''

    if args.huber_k !=1:
        extra_save += '_'+str(args.huber_k)

    if 'full_generations' in args.data:
        extra_save += '_'+'prompt_enc'
    ## TODO: DO I NEED TO CONSIDER THE PROMPT?

    if args.emotion != 'None':
        extra_save += '_'+args.emotion

    if args.emotion != 'None':
        args.more_balanced = 'True'

    if '+' in args.emotion:
        args.emotion_set = args.emotion.split('+')
    else:
        args.emotion_set = [args.emotion]

    if args.results_folder == 'None':
        results_folder = Path(args.data).parent
    else:
        results_folder = Path(args.results_folder)

    if args.mdp_mode:
        args.save_path = results_folder / ('quantile_learner_mdp2'+extra_save) / 'quantile_learner_mdp.pkl'
        args.log_path  = args.save_path.parent  / 'log_quantile_learner_mdp.pkl'
    else:
        args.save_path = results_folder / ('quantile_learner'+extra_save) /  'quantile_learner.pkl'
        args.log_path  = args.save_path.parent / 'log_quantile_learner.pkl'

    print(f'saving to : {args.save_path}')

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    set_seeds(seed = args.seed)

    if not args.mdp_mode:
        # load lm model for calculating hidden states
        config = GPT2Config.from_pretrained(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|generate|>","<|score|>"
        ]})
        model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

    else:
        model = None; tokenizer=None

    # get data
    train_data, state_dim = prepare_data(args, tokenizer, split='train', subset=args.subset)
    eval_data, state_dim_eval, eval_dataset = prepare_data(args, tokenizer, split='validation', subset=args.subset)
    eval_df = pd.DataFrame(eval_dataset)

    if model is not None:
        state_dim = model.config.n_embd

    # set up TD learner
    n_quantiles = args.n_quantiles
    gamma = 1.0 #0.99
    hidden_dim=None if args.linear else args.hidden_dim
    Z_network = TD_Learner(state_dim, n_quantiles, hidden_dim).to(device)
    Z_network_tgt = TD_Learner(state_dim, n_quantiles, hidden_dim).to(device)
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)

    if args.n_quantiles==12:
        taus = (2 * np.arange(10) + 1) / (2.0 * n_quantiles)
        taus=np.append(taus, 0.99)
        taus=np.insert(taus, 0, 0.01)

    tau = torch.Tensor(taus).view(1, 1, -1).to(device) # third dimensional will be quantiles

    # set up optimizers, and learning rate schedule
    optimizer = torch.optim.Adam(params=Z_network.parameters(), lr=args.learning_rate)

    log_dict = {}
    print(f'number of batches in one epoch {len(train_data)}')

    for epoch in range(1, args.epochs):

        epoch_loss = 0

        for idx, (batch, text) in tqdm(enumerate(train_data, start=1), leave=True, position=0, total=len(train_data)):

            states, mask, rewards, input_ids = calc_state_from_batch(batch, device, model, mdp_mode=args.mdp_mode, emotion=args.emotion)

            optimizer.zero_grad()

            # current value (distributions); (batch_size x sequence_size x number of quantiles)
            thetas = Z_network(states)
            thetas = thetas*mask # zero out for padding tokens

            # next state value (distributions); (batch_size x sequence_size x number of quantiles)
            tgt_dist = Z_network_tgt(states).detach() # so you don't pass the gradients through here.
            tgt_dist = torch.roll(tgt_dist, shifts=-1, dims=1) # actually turn it into next state
            tgt_dist = tgt_dist*mask

            # compute targets
            #   i.e. set reward for last token in sequence_size
            #   and zero out the last state values (replace with done matrix)
            reward_tensor = torch.zeros(thetas.shape).to(device)
            for i in range(reward_tensor.shape[0]):
                last_tok_idx = int(torch.argmax(mask[i,:].t()*torch.arange(mask.shape[1]).to(device))) # arange up to max length sequence, multiply by the mask
                reward_tensor[i,last_tok_idx,:]=rewards[i] # unnecessary for terminal rewards only
                tgt_dist[i,last_tok_idx,:]=0.

            target_thetas = reward_tensor + gamma*tgt_dist

            # compute quantile or normal td loss
            if n_quantiles==1:
                loss = (target_thetas - thetas)**2
            else:
                # Notes: first convert thetas to (states x quantiles), and target thetas to (quantiles x states x 1)
                # collapsed across batch and seq; the result will be (quantiles x states x quantiles)
                diff = target_thetas.view(-1, n_quantiles).t().unsqueeze(-1) - thetas.view(-1, n_quantiles)
                loss = huber(diff, args.huber_k) * (tau - (diff.detach() < 0).float()).abs()

            loss = loss.mean()
            loss.backward() # consider the grad clipping in the main.py for the DQN.
            optimizer.step()

            epoch_loss+=loss.detach().cpu().numpy()

            # switch target network
            if idx % args.target_every==0 and idx !=0:
               Z_network_tgt.load_state_dict(Z_network.state_dict())

        # log every epoch
        print(f'\nepoch{epoch} loss:{epoch_loss / idx:.3f}')
        log_dict = append_to_log(log_dict, 'loss', epoch_loss / idx)
        log_dict = append_to_log(log_dict, 'epoch', epoch)

        # printing for checking its progress
        if args.mdp_mode:
            for state in range(state_dim):
                state_vec = np.zeros(state_dim)
                state_vec[state]=1
                with torch.no_grad():
                    theta_hats = Z_network(torch.Tensor(state_vec).to(device)).detach().cpu().numpy().round(2)
                print(f' state {state} theta_hats: {theta_hats}')
                log_dict = append_to_log(log_dict, f'state {state}', theta_hats)
        else:

            examples = []

            # first get examples from the type of emotion (from validation set)
            #emotion_set = args.emotion_set
            emotion_set = ['anger','annoyance','sadness','grief','fear','nervousness', 'admiration','optimism']

            for emotion in emotion_set:
                #eval_df.sort_values(by=emotion, ascending=False).iloc[0:10][['text', emotion]] # to inspect
                examples.extend(list(eval_df.sort_values(by=emotion, ascending=False).iloc[0:10]['text'].values))

            # make predictions on the examples
            for example in examples:
                input = tokenizer(example, return_tensors='pt').to(device)
                with torch.no_grad():
                    output = model(input_ids=input['input_ids'],
                                   attention_mask=input['attention_mask'],
                                   output_hidden_states=True)
                    states = output['hidden_states'][-1]
                    theta_hats = Z_network(states).detach().cpu().numpy().round(2)
                    theta_hats_last = theta_hats[:,-1,:].squeeze()
                print(f' example: {example}')
                print(f' theta_hats: {theta_hats_last}')
                log_dict = append_to_log(log_dict, example, theta_hats_last)
                log_dict = append_to_log(log_dict, 'exnd_'+example, theta_hats)

            # now, what about a more quantitative measure of performance?
            # TO-DO:
            # One option is to look at the mean; that's important; but then I'll also need a metric of the distribution

        # save the model and the log
        if (epoch) % 1 == 0 and epoch !=0:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            fileend = f'_epoch{epoch}.pkl'
            torch.save(Z_network.state_dict(), str(args.save_path).replace('.pkl', fileend))
            pickle.dump(log_dict, open(str(args.log_path).replace('.pkl', fileend), "wb" ))

if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 25 --mdp_mode
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 25
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 200 --filter "I am" --batch_size 40 --linear
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 1000 --filter "I" --batch_size 40 --hidden_dim 100

    # MDP runs
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 50 --batch_size 40 --hidden_dim 101 --mdp_mode --n_quantiles 20 --target_every 100 --learning_rate 1e-3 --huber_k 0.1

    # Dataset v3
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 40 --hidden_dim 101 --n_quantiles 10 --target_every 10 --huber_k 0.1
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 40 --hidden_dim 101 --n_quantiles 10 --target_every 10 --huber_k 0.1
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 102 --n_quantiles 10 --target_every 10 --huber_k 0.1
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 103 --n_quantiles 10 --target_every 10 --huber_k 0.1
    # CUDA_VISIBLE_DEVICES=0 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3/full_generations.txt'

    # Dataset v4
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 20 --target_every 10 --huber_k 0.1 --filter_out 'You'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 101 --n_quantiles 10 --target_every 10 --huber_k 0.1 --filter_out 'You'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 102 --n_quantiles 10 --target_every 20 --huber_k 0.1 --filter_out 'You'
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 20 --hidden_dim 103 --n_quantiles 10 --target_every 5 --huber_k 0.1 --filter_out 'You'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --filter_out 'You' --learning_rate 1e-4     **this seems to be working the best

    # this is working
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 2 --hidden_dim 105 --n_quantiles 10 --target_every 20 --huber_k 0.1 --filter 'I woke up. I ate breakfast. And I' --learning_rate 1e-4
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 105 --n_quantiles 10 --target_every 2 --huber_k 0.1 --filter 'I woke up. I ate breakfast. And I' --learning_rate 1e-4

    # this is working
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 2 --hidden_dim 105 --n_quantiles 10 --target_every 20 --huber_k 0.1 --filter 'I woke up. I ate breakfast. And' --learning_rate 1e-4

    # can run later
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 2 --hidden_dim 105 --n_quantiles 10 --target_every 20 --huber_k 0.1 --filter 'I woke up. I ate breakfast.' --learning_rate 1e-4

    # Dataset v5
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_5/full_generations.txt'
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_5/ends.txt'

    # Dataset v6
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 50 --batch_size 10 --hidden_dim 100 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 2e-5 --data 'data/results/single_sentences_IYou_6/full_generations.txt'
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 50 --batch_size 10 --hidden_dim 10 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 2e-5 --data 'data/results/single_sentences_IYou_6/full_generations.txt'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 50 --batch_size 10 --linear --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 2e-5 --data 'data/results/single_sentences_IYou_6/full_generations.txt'

    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 10 --batch_size 10 --linear --n_quantiles 12 --target_every 10 --huber_k 0.01 --learning_rate 2e-5 --data 'data/results/single_sentences_IYou_6/full_generations.txt'


    # Dataset v3 with emotions
    # CUDA_VISIBLE_DEVICES=1 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'anger+annoyance'
    # CUDA_VISIBLE_DEVICES=0 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'sadness+grief'
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'fear+nervousness'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt'

    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 50 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'admiration'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 50 --batch_size 10 --hidden_dim 104 --n_quantiles 10 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'optimism'

    ## running linear
    # CUDA_VISIBLE_DEVICES=0 python train_rl_batch.py --epochs 20 --batch_size 10 --linear --n_quantiles 12 --target_every 5 --huber_k 0.1 --learning_rate 1e-4 --data 'data/results/single_sentences_IYou_3_emo/cmbnd_full_generations_w_emotions_shfld.txt' --emotion 'sadness+grief'


    ## running with posts dataset
    # CUDA_VISIBLE_DEVICES=0 python train_rl_batch.py
