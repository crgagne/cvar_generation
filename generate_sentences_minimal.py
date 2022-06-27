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
    GPTJForCausalLM,
    GPT2Tokenizer,
    OPTForCausalLM,
)

import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm

from generator import generate


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook-opt-2.7B")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default='single_sentences_test')
    parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--max_length", type=int, default=20)
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
    if 'gpt-2' in args.model:
        config = GPT2Config.from_pretrained(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
        model.config.pad_token_id = model.config.eos_token_id
    elif 'gpt-j' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16) # optional to include this precision
        if args.gpus==0: # haven't tried yet
            model = GPTJForCausalLM.from_pretrained(args.model, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    elif 'opt' in args.model:
        model = OPTForCausalLM.from_pretrained(args.model)# .half() # trying temporarily # torch_dtype=torch.float16 # generates garbage
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    model.to(device)

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


        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # generate possible continuations
        output, _ = generate(model, tokenizer,
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=inputs['input_ids'].shape[1]+args.max_length, num_beams = 1,
                            temperature=1, num_return_sequences=10,
                            do_sample=True, eos_token_id=13,
                            top_k = args.top_k, top_p=0.95,
                            allowed_word_ids = None,
                            )
        # output = model.generate(input_ids=inputs['input_ids'],
        #                     attention_mask=inputs['attention_mask'],
        #                     max_length=inputs['input_ids'].shape[1]+args.max_length, num_beams = 1,
        #                     temperature=1, num_return_sequences=10,
        #                     do_sample=True, eos_token_id=13,
        #                     top_k = args.top_k, top_p=0.95,
        #                     allowed_word_ids = None)

        decoded.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        prompt_storage.extend([prompt for _ in range(10)])

    print(decoded)

if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_minimal.py  --model 'models/pretrained/facebook-opt-2.7b' --num_iterations 10 --gpus 1
