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

import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm
import pickle

from generator import generate

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
    parser.add_argument("--save_folder", type=str, default='single_sentences_IYou_2')
    parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--example_list", type=str, default="ood_examples.txt")
    parser.add_argument("--max_length", type=int, default=20)
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
    model.to(device)

    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    # load prompts
    f = open(Path(args.save_folder) / args.prompt_list, "r")
    prompts = f.readlines()
    prompts = [prompt.replace('\n','') for prompt in prompts]

    f = open(Path(args.save_folder) / args.example_list, "r")
    examples = f.readlines()
    examples = [example.replace('\n','') for example in examples]

    outputs = {}
    for example in tqdm(examples):

        decoded = []
        for i in tqdm(range(args.num_iterations)):

            # choose 3 random prompts
            prompt = ' '.join(np.random.choice(prompts, size=3))

            if np.random.binomial(n=1,p=0.5)==1:
                prompt = prompt.replace('I', 'You')

            prompt = prompt+' '+example

            # tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # generate possible continuations
            output, _ = generate(model, tokenizer,
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_length=inputs['input_ids'].shape[1]+args.max_length, num_beams = 1,
                                temperature=1, num_return_sequences=10,
                                do_sample=True, eos_token_id=13,
                                bad_words_ids = None,
                                top_k = 0, top_p=0.95,
                                allowed_word_ids = None,
                                )

            decoded.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

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
            end = d.split('.')[3].strip()+'.'
            ends.append(end)

        rewards = []
        for i in range(0,len(ends),20):
            j = np.min([i+20,len(ends)])
            rewards.extend(score_sentiment(ends[i:j], sentiment_tokenizer, sentiment_model, device))
        assert len(rewards)==len(ends)

        sort_idx = np.argsort(rewards)
        rewards = [rewards[i] for i in sort_idx]
        decoded = [decoded[i] for i in sort_idx]
        ends = [ends[i] for i in sort_idx]

        outputs[example]={}
        outputs[example]['decoded']=decoded
        outputs[example]['rewards']=rewards
        outputs[example]['ends']=ends

    filename = Path(args.save_folder) / f'odd_examples_output.pkl'
    pickle.dump(outputs, open(filename, 'wb'))

if __name__ == '__main__':

    main()


    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_examples.py  --model models/pretrained/gpt2-large --num_iterations 10
