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
import torch.nn.functional as functional
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import pickle


# from torch.utils.data.dataloader import default_collate this doesn't work with hugging face dataset
from transformers import default_data_collator
from batch_datasets import get_batch_dataset
from rl_learner import TD_Learner

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
    # consider nn.functional.HuberLoss(diff) # reduction = None

def prepare_data(args, tokenizer, n_rewards=3):

    # load dataset
    dataset = get_batch_dataset(args.data, n_rewards=n_rewards)

    # tokenize
    def tokenize(batch):
       return tokenizer(batch['text'], truncation=True, max_length=args.max_length, padding='max_length')
    dataset = dataset.map(tokenize, num_proc=1, batched=True)

    if args.filter is not None:
        dataset = dataset.filter(lambda example: example['text'].startswith(args.filter))

    # tensorize sentiment
    def tensorize(batch):
        batch['summed_reward']=[torch.Tensor([summed_reward]) for summed_reward in batch['summed_reward']]
        if 'reward' in batch:
            batch['reward']=[torch.Tensor([reward]) for reward in batch['reward']]
        if 'composite_reward' in batch:
            batch['composite_reward']=[torch.Tensor([reward]) for reward in batch['composite_reward']]
        return(batch)
    dataset = dataset.map(tensorize, num_proc=1, batched=True)

    # filter potentially #
    if args.filter_out is not None:
        dataset = dataset.filter(lambda example: not example['text'].startswith(args.filter_out))

    # batch data
    def collate_with_strings(batch, str_column = 'text'):
        new_batch = []; strings = []
        for _batch in batch:
            strings.append(_batch[str_column])
            _batch.pop(str_column, None)
            new_batch.append(_batch)
        try:
            coll_batch = default_data_collator(new_batch)
        except:
            import ipdb; ipdb.set_trace()
        return coll_batch, strings
    train_data = DataLoader(dataset, collate_fn=collate_with_strings, batch_size=args.batch_size, shuffle=True)
    state_dim = None
    # batch = next(iter(train_data))

    return(train_data, state_dim)

def average_states_by_period(states, mask, input_ids, device, n_periods=5, period_tok_id=13, pad_tok_id=50256):

    avg_states = torch.zeros((states.shape[0], n_periods, states.shape[2])).to(device)
    avg_mask = torch.zeros((mask.shape[0], n_periods, mask.shape[2])).to(device)
    for idx in range(input_ids.shape[0]):
        period_locs = np.where(input_ids[idx,:].detach().cpu().numpy()==period_tok_id)[0]
        period_locs = np.insert(period_locs,0,0)
        assert (int(input_ids[idx,-1].cpu())==pad_tok_id) or (int(input_ids[idx,-1].cpu())==period_tok_id) # either ends with a pad or a period
        for i in range(len(period_locs)-1):
            avg_states[idx,i,:] = torch.mean(states[idx,period_locs[i]:period_locs[i+1],:],dim=0)
            max, _ = torch.max(mask[idx,period_locs[i]:period_locs[i+1],:], dim=0) # using the max here shouldn't matter they should all be 1's
            avg_mask[idx,i,:] = max

    assert avg_states.shape[1]==n_periods
    return(avg_states, avg_mask)


def calc_state_from_batch(batch, device, model, mdp_mode=False, n_periods=5, intermediate_rewards=False, composite_reward=False):

    assert intermediate_rewards==False or composite_reward==False

    if mdp_mode:
        states = batch[0].to(device)
        rewards = batch[1].to(device)
        mask = batch[2].unsqueeze(-1).to(device)

    else:
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        if intermediate_rewards and not composite_reward:
            rewards = batch['rewards'].to(device)
            rewards = functional.pad(rewards, (n_periods-rewards.shape[1], 0), mode='constant', value=0) # pad in front for prompt
        else:
            rewards = batch['summed_reward'].to(device)

        if composite_reward:
            rewards = batch['composite_reward'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=mask,
                           output_hidden_states=True)

        # feed into TD learenr
        states = output['hidden_states'][-1]
        mask = mask.unsqueeze(-1)

        # average for each sentence
        states, mask = average_states_by_period(states, mask, input_ids, device,
                                                n_periods=n_periods, period_tok_id=13, pad_tok_id=50256)

    return(states, mask, rewards)

def append_to_log(log_dict, key, value):
    if key not in log_dict.keys():
        log_dict[key]=[value]
    else:
        log_dict[key].append(value)
    return(log_dict)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2-large")
    parser.add_argument("--data", type=str, default='data/results/sentence_chains_I_2/generations_tmp.txt')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_quantiles", type=int, default=10)
    parser.add_argument("--mdp_mode", action='store_true')
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--hidden_dim", type=int, default=102)
    parser.add_argument("--target_every", type=int, default=10)
    parser.add_argument("--huber_k", type=float, default=0.1)
    parser.add_argument("--filter_out", type=str, default=None)
    parser.add_argument("--use_nt_rewards", type=str, default='False')
    parser.add_argument("--use_composite_rewards", type=str, default='False')
    parser.add_argument("--split_questions", type=str, default='False')
    parser.add_argument("--n_rewards", type=int, default=3)

    args = parser.parse_args()
    extra_save = '_linear' if args.linear else '_'+str(args.hidden_dim)
    extra_save += '_'+args.filter.replace(' ','_') if args.filter is not None else ''
    if args.huber_k !=1:
        extra_save += '_'+str(args.huber_k)
    if eval(args.use_nt_rewards):
        extra_save += '_nt_rewards'
    if eval(args.use_composite_rewards):
        extra_save += '_composite_rewards'


    if args.mdp_mode:
        args.save_path = Path(args.data).parent / ('quantile_learner_mdp2'+extra_save) / 'quantile_learner_mdp.pkl'
        args.log_path  = args.save_path.parent  / 'log_quantile_learner_mdp.pkl'
    else:
        args.save_path = Path(args.data).parent / ('quantile_learner'+extra_save) /  'quantile_learner.pkl'
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
        model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
        model.to(device)

    else:
        model = None; tokenizer=None

    # get data
    train_data, state_dim = prepare_data(args, tokenizer, n_rewards=args.n_rewards)

    if model is not None:
        state_dim = model.config.n_embd

    # set up TD learner
    n_quantiles = args.n_quantiles
    gamma = 1.0 #0.99
    hidden_dim=None if args.linear else args.hidden_dim
    Z_network = TD_Learner(state_dim, n_quantiles, hidden_dim).to(device)
    Z_network_tgt = TD_Learner(state_dim, n_quantiles, hidden_dim).to(device)
    tau = torch.Tensor((2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)).view(1, 1, -1).to(device) # third dimensional will be quantiles

    # set up optimizers, and learning rate schedule
    optimizer = torch.optim.Adam(params=Z_network.parameters(), lr=args.learning_rate)

    log_dict = {}
    print(f'number of batches in one epoch {len(train_data)}')

    for epoch in range(1, args.epochs):

        epoch_loss = 0

        for idx, (batch, text) in tqdm(enumerate(train_data, start=1), leave=True, position=0):

            if eval(args.split_questions):
                n_periods = text[0].count('.')+text[0].count('?')
                raise NotImplementedError
            else:
                n_periods = text[0].count('.')

            states, mask, rewards = calc_state_from_batch(batch, device, model, mdp_mode=args.mdp_mode, n_periods=n_periods,
                                                          intermediate_rewards=eval(args.use_nt_rewards), composite_reward=eval(args.use_composite_rewards))

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
            for i in range(reward_tensor.shape[0]): # looping over batch size
                last_tok_idx = int(torch.argmax(mask[i,:].t()*torch.arange(mask.shape[1]).to(device))) # arange up to max length sequence, multiply by the mask
                tgt_dist[i,last_tok_idx,:]=0.
                if eval(args.use_nt_rewards):
                    reward_tensor[i,:,:]=rewards[i].unsqueeze(-1)
                else: # terminal rewards
                    reward_tensor[i,last_tok_idx,:]=rewards[i] # unnecessary for terminal rewards only

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

            examples = ['I puked all over my living room floor and waited.',
                        "You're dirty.",
                        'I took the stupid potty training class and passed out on the mat.',
                        'My knuckles are tired.',
                        'I got on a park bench and happily listened to Mozart and Praxis performed.',
                        'I was, of course, all six packer eyes.',
                        "That was his New Year's Resolution: health.",
                        "It was very nice.",
                        "The benefits of moving early always seem pretty obvious to me.",
                        "The sun was shining and I'd made it to campus on time."]
            for example in examples:
                input = tokenizer(example, return_tensors='pt').to(device)
                input_ids = input['input_ids']
                mask = input['attention_mask']
                with torch.no_grad():
                    output = model(input_ids=input_ids,
                                   attention_mask=mask,
                                   output_hidden_states=True)
                    states = output['hidden_states'][-1]

                    states, _ = average_states_by_period(states, mask.unsqueeze(-1), input_ids, device,
                                                            n_periods=example.count('.'), period_tok_id=13, pad_tok_id=50256)

                    theta_hats = Z_network(states).detach().cpu().numpy().round(2)
                    theta_hats_last = theta_hats[:,-1,:].squeeze()
                print(f' example: {example}')
                print(f' theta_hats: {theta_hats_last}')
                log_dict = append_to_log(log_dict, example, theta_hats_last)

        # save the model and the log
        if (epoch) % 1 == 0 and epoch !=0:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            fileend = f'_epoch{epoch}.pkl'
            torch.save(Z_network.state_dict(), str(args.save_path).replace('.pkl', fileend))
            pickle.dump(log_dict, open(str(args.log_path).replace('.pkl', fileend), "wb" ))

if __name__ == '__main__':

    main()

    # Chains v2 v3
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_2/generations_using.txt'
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_3/generations_using.txt'

    # Chains v4
    # Running with terminal rewards
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_4/generations_using.txt'

    # Running with intermediate rewards
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_4/generations_using.txt' --use_nt_rewards 'True'

    # Chains v5
    # Running with composite rewards only (small learning rate)
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch_sentence_chains.py --epochs 20 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using.txt' --use_composite_rewards 'True' --learning_rate 5e-5
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch_sentence_chains.py --epochs 20 --batch_size 20 --hidden_dim 10 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using.txt' --use_composite_rewards 'True' --learning_rate 5e-5

    # (faster learning rate)
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch_sentence_chains.py --epochs 50 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using.txt' --use_composite_rewards 'True' --learning_rate 1e-3
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch_sentence_chains.py --epochs 50 --batch_size 20 --hidden_dim 10 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using.txt' --use_composite_rewards 'True' --learning_rate 1e-3

    # (faster learning rate; more data (generations 2)
    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 100 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using2.txt' --use_composite_rewards 'True' --learning_rate 1e-3
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch_sentence_chains.py --epochs 100 --batch_size 20 --hidden_dim 10 --n_quantiles 10 --target_every 20 --huber_k 0.1 --data 'data/results/sentence_chains_I_5/generations_using2.txt' --use_composite_rewards 'True' --learning_rate 1e-3
