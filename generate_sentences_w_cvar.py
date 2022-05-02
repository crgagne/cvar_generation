from pathlib import Path
import argparse
from helpers import set_seeds
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from models import GPT2CustomDoubleHeadsModel
import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm

from generator import generate
from rl_learner import TD_Learner
import pickle


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def score_sentiment(sentences, sentiment_tokenizer, sentiment_model, device):

    output = sentiment_model(**sentiment_tokenizer(sentences, return_tensors='pt', padding=True).to(device))
    probs = softmax(output[0].detach().cpu().numpy(),axis=1)

    if probs.shape[1]==5:
        scores = np.dot(probs,np.arange(-2,3))
    else:
        scores = np.dot(probs,np.arange(-1,2))

    #scores = np.argmax(probs,axis=1)
    sort_idx = np.argsort(scores)
    scores_sorted = [scores[i] for i in sort_idx]
    sentences_sorted = [sentences[i] for i in sort_idx]

    return(scores)

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2-large")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default='single_sentences_I_1')
    parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--restrict_w_data", action='store_true')
    parser.add_argument("--alpha", type=float, default=0.2)
    #parser.add_argument("--n_return", type=int, default=10) # not implemented
    args = parser.parse_args()
    args.save_folder = Path(__file__).parent / 'data' / 'results' / args.save_folder
    print(f'save folder: {args.save_folder}')

    set_seeds(seed = args.seed)

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model
    config = GPT2Config.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    if args.restrict_w_data:
        file = args.save_folder / 'round1_ends.txt'
        f = open(file, "r")
        sentences = f.readlines()
        sentences = [s.replace('\n','') for s in sentences]
        sentences = [s.split('r=')[0] for s in sentences]
        tokenized_sentences = tokenizer(sentences, return_tensors='pt', padding=True, max_length=args.max_length)['input_ids']
        tokenized_sentences = tokenized_sentences.to(device)

    else:
        tokenized_sentences = None

    # load cvar value model
    n_quantiles = 5; hidden_dim = 100
    learning_filename=args.save_folder / 'quantile_learner_100' / 'quantile_learner_epoch200.pkl'
    Z_network = TD_Learner(config.n_embd, n_quantiles, hidden_dim=hidden_dim).to(device)
    Z_network.load_state_dict(torch.load(learning_filename))

    # load prompts
    f = open(Path(args.save_folder) / args.prompt_list, "r")
    prompts = f.readlines()
    prompts = [prompt.replace('\n','') for prompt in prompts]
    decoded = []
    alphas = []
    p_storage = []
    pd_storage = []

    for i in tqdm(range(args.num_iterations)):

        # choose 3 random prompts
        prompt = ' '.join(np.random.choice(prompts, size=3))

        if np.random.binomial(n=1,p=0.5)==1:
            prompt = prompt.replace('I', 'You')

        #prompt = 'I'
        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # generate possible continuations
        output, other_outputs = generate(model, tokenizer,
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=inputs['input_ids'].shape[1]+args.max_length, num_beams = 1,
                            temperature=1, num_return_sequences=1,
                            do_sample=True, eos_token_id=13,
                            bad_words_ids = None,
                            top_k = 10, top_p=0.95,
                            allowed_word_ids = None,
                            data_to_restrict_w = tokenized_sentences,
                            cvar_alpha = args.alpha,
                            Z_network = Z_network,
                            tokenized_prompt = tokenized_prompt,
                            return_dict_in_generate=False,
                            )

        decoded.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

        alphas.append(other_outputs['alphas'])
        p_storage.append(other_outputs['p_storage'])
        pd_storage.append(other_outputs['pd_storage'])

    # process a bit
    decoded_proc =[]
    for d in decoded:
        include=True
        d = d.replace('\n','').replace('\r','').replace('"','')
        d = d.replace('[','').replace(']','')
        if d[-1]!='.':
            include=False
        if '?' in d:
            include=False
        if has_numbers(d):
            include=False
        num_words = d.split(' ')
        if len(num_words)<3:
            include=False

        if include:
            decoded_proc.append(d)

    decoded = decoded_proc

    ends = []
    for d in decoded:
        ends.append(d.split('.')[-2].strip()+'.')


    rewards = []
    for i in range(0,len(ends),20):
        j = np.min([i+20,len(ends)])
        rewards.extend(score_sentiment(ends[i:j], sentiment_tokenizer, sentiment_model, device))
    assert len(rewards)==len(ends)

    sort_idx = np.argsort(rewards)
    rewards = [rewards[i] for i in sort_idx]
    decoded = [decoded[i] for i in sort_idx]
    ends = [ends[i] for i in sort_idx]

    # save results
    if args.restrict_w_data:
        filename1 = Path(args.save_folder) / f'round1_full_generations_cvar_{args.alpha}.txt'
        filename2 = Path(args.save_folder) / f'round1_ends_cvar_{args.alpha}.txt'
    else:
        filename1 = Path(args.save_folder) / f'round1_full_generations_cvar_{args.alpha}_unres.txt'
        filename2 = Path(args.save_folder) / f'round1_ends_cvar_{args.alpha}_unres.txt'

    with open(filename1, 'w') as f1, open(filename2, 'w') as f2:
        for gen, end, r in zip(decoded, ends, rewards):
            line = f"{gen} r={r:.3f}\n"
            print(line)
            f1.write(line)
            line = f"{end} r={r:.3f}\n"
            f2.write(line)

    other_outputs = {}
    other_outputs['alphas']=alphas
    other_outputs['p_storage']=p_storage
    other_outputs['pd_storage']=pd_storage
    if args.restrict_w_data:
        filename3 = Path(args.save_folder) / f'cvar_output_{args.alpha}.pkl'
    else:
        filename3 = Path(args.save_folder) / f'cvar_output_{args.alpha}_unres.pkl'

    pickle.dump(other_outputs, open(filename3, 'wb'))


if __name__ == '__main__':

    main()


    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --restrict_w_data --alpha 0.2
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --restrict_w_data --alpha 0.5
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --restrict_w_data --alpha 0.05

    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 
