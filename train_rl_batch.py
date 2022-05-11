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

# from torch.utils.data.dataloader import default_collate this doesn't work with hugging face dataset
from transformers import default_data_collator
from batch_datasets import get_batch_dataset
from rl_learner import TD_Learner

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
    # consider nn.functional.HuberLoss(diff) # reduction = None

def prepare_data(args, tokenizer):

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
        dataset = get_batch_dataset(args.data)

        # tokenize
        def tokenize(batch):
           return tokenizer(batch['text'], truncation=True, max_length=args.max_length, padding='max_length')
        dataset = dataset.map(tokenize, num_proc=1, batched=True)

        if args.filter is not None:
            dataset = dataset.filter(lambda example: example['text'].startswith(args.filter))

        # TODO: consider replacing the '.' 13 token with the pad token, so that it gets assigned to the reward

        # tensorize sentiment
        def tensorize(batch):
            batch['sentiment']=[torch.Tensor([sentiment]) for sentiment in batch['sentiment']]
            return(batch)
        dataset = dataset.map(tensorize, num_proc=1, batched=True)

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

    return(train_data, state_dim)

def calc_state_from_batch(batch, device, model, mdp_mode=False):

    if mdp_mode:
        states = batch[0].to(device)
        rewards = batch[1].to(device)
        mask = batch[2].unsqueeze(-1).to(device)

    else:
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        rewards = batch['sentiment'].to(device)

        # TODO: consider precomputing hidden states
        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=mask,
                           output_hidden_states=True)

        # feed into TD learenr
        states = output['hidden_states'][-1]
        mask = mask.unsqueeze(-1)
        assert len(states.shape)==len(mask.shape)

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
    parser.add_argument("--data", type=str, default='data/results/single_sentences_IYou_2/ends.txt')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_quantiles", type=int, default=10)
    parser.add_argument("--mdp_mode", action='store_true')
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--target_every", type=int, default=10)
    parser.add_argument("--huber_k", type=float, default=0.1)

    args = parser.parse_args()
    extra_save = '_linear' if args.linear else '_'+str(args.hidden_dim)
    extra_save += '_'+args.filter.replace(' ','_') if args.filter is not None else ''
    if args.huber_k !=1:
        extra_save += '_'+str(args.huber_k)

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
    train_data, state_dim = prepare_data(args, tokenizer)
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

    for epoch in range(1, args.epochs):

        epoch_loss = 0

        for idx, (batch, text) in tqdm(enumerate(train_data, start=1), leave=True, position=0):

            states, mask, rewards = calc_state_from_batch(batch, device, model, mdp_mode=args.mdp_mode)

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
            examples = ['I am so stupid to have this failed.',
                        'I am amazing and a great speaker.',
                        'I am so stupid',
                        'I am amazing',
                        'I am']
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

        # save the model and the log
        if (epoch) % 5 == 0 and epoch !=0:
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

    # New runs
    # CUDA_VISIBLE_DEVICES=2 python train_rl_batch.py --epochs 100 --batch_size 40 --hidden_dim 101 --n_quantiles 10 --target_every 10 --huber_k 0.1

    # CUDA_VISIBLE_DEVICES=3 python train_rl_batch.py --epochs 100 --batch_size 40 --hidden_dim 101 --n_quantiles 10 --target_every 10 --huber_k 0.1
