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
    AutoModelForCausalLM,
    GPTJForCausalLM
)

import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm

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
    #parser.add_argument("--model", type=str, default="models/finetuned/gpt2-large/SMHD_posts_depctrl_v1/checkpoint-2400")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default='single_sentences_longer_reddit_v1')
    parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--comp_prompts", action='store_true')
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    args.save_folder = Path(__file__).parent / 'data' / 'results' / args.save_folder
    print(f'save folder: {args.save_folder}')

    set_seeds(seed = args.seed)

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model
    if 'gpt2' in args.model:
        # load model add new tokens
        config = GPT2Config.from_pretrained(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|generate|>","<|score|>","<|pad|>"
        ]})
        tokenizer.pad_token = tokenizer.encode("<|pad|>")
        model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
        model.resize_token_embeddings(len(tokenizer))

    eos_token_id = 13 # for periods
    #eos_token_id = 50258 # for score

    model.to(device)

    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    # load prompts
    f = open(Path(args.save_folder) / args.prompt_list, "r")
    prompts = f.readlines()
    prompts = [prompt.replace('\n','') for prompt in prompts]
    decoded = []
    prompt_storage = []
    for i in tqdm(range(args.num_iterations)):

        # choose 3 random prompts
        if args.comp_prompts:
            prompt = ' '.join(np.random.choice(prompts, size=3))
        else:
            prompt =  np.random.choice(prompts, size=1)[0]

        #if np.random.binomial(n=1, p=0.5)==1:
        #    prompt = prompt.replace('I', 'You')
        if 'finetuned' in args.model:
            prompt = '<|generate|>'+prompt

        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        if 'gpt-2' in args.model:
            #bad_words = ["\n", "\r", '"']
            bad_words = ["\n", "\r", '"',' "',' (',' )',' [',' ]','."','?"',',"']
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
        else:
            bad_words = ["\n", "\r", '"']
            bad_words_ids = [tokenizer.encode(bad_word) for bad_word in bad_words]

        # generate possible continuations
        output, _ = generate(model, tokenizer,
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=inputs['input_ids'].shape[1]+args.max_length,
                            num_beams = 1,
                            temperature=1,
                            num_return_sequences=10,
                            do_sample=True,
                            eos_token_id=eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            bad_words_ids = bad_words_ids,
                            top_k = args.top_k,
                            top_p=0.95,
                            allowed_word_ids = None,
                            )

        decoded.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        prompt_storage.extend([prompt for _ in range(10)])

    # process a bit
    decoded_proc =[]
    prompt_storage2 = []
    for d, pr in zip(decoded, prompt_storage):
        include=True
        d = d.replace('\n','').replace('\r','').replace('"','')
        d = d.replace('[','').replace(']','')
        #if d[-1]!='.':
        #    include=False
        #if '?' in d:
        #    include=False
        #if has_numbers(d):
        #    include=False
        num_words = d.split(' ')
        if len(num_words)<3:
            include=False

        if include:
            decoded_proc.append(d)
            prompt_storage2.append(pr)

    decoded = decoded_proc
    prompt_storage = prompt_storage2
    assert len(decoded)==len(prompt_storage)

    ends = []
    for d,p in zip(decoded, prompt_storage):
        ends.append(d.replace(p,'').strip())

    rewards = []
    for i in range(0,len(ends),20): # 20 is a batch size
        j = np.min([i+20,len(ends)])
        rewards.extend(score_sentiment(ends[i:j], sentiment_tokenizer, sentiment_model, device))
    assert len(rewards)==len(ends)

    sort_idx = np.argsort(rewards)
    rewards = [rewards[i] for i in sort_idx]
    decoded = [decoded[i] for i in sort_idx]
    ends = [ends[i] for i in sort_idx]

    # save results
    if 'finetuned' in args.model:
        filename1 = Path(args.save_folder) / 'full_generations_ft.txt'
        filename2 = Path(args.save_folder) / 'ends_ft.txt'
    else:
        filename1 = Path(args.save_folder) / 'full_generations.txt'
        filename2 = Path(args.save_folder) / 'ends.txt'
    with open(filename1, 'w') as f1, open(filename2, 'w') as f2:
        for gen, end, r in zip(decoded, ends, rewards):
            line = f"{gen} r={r:.3f}\n"
            print(line)
            f1.write(line)
            line = f"{end} r={r:.3f}\n"
            f2.write(line)


if __name__ == '__main__':

    main()


    # CUDA_VISIBLE_DEVICES=2 python generate_sentences.py  --model models/pretrained/gpt2-large --num_iterations 5000
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences.py  --model models/pretrained/gpt2-large --num_iterations 1000

    # CUDA_VISIBLE_DEVICES=3 python generate_sentences.py  --model models/pretrained/EleutherAI-gpt-j-6B --num_iterations 10 --gpus 0

    # comparing finetuned gpt-2 to not ..
    # CUDA_VISIBLE_DEVICES=1 python generate_sentences.py  --model models/finetuned/gpt2-large/SMHD_posts_depctrl_v1/checkpoint-2400 --num_iterations 10 --gpus 1
    # CUDA_VISIBLE_DEVICES=1 python generate_sentences.py  --model models/pretrained/gpt2-large --num_iterations 10000 --gpus 1
