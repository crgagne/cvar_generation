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
    #parser.add_argument("--value_model", type=str, default="quantile_learner_101_0.1/quantile_learner_epoch95.pkl")
    parser.add_argument("--value_model", type=str, default="quantile_learner_102_0.1/quantile_learner_epoch10.pkl")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default='single_sentences_IYou_3')
    parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--restrict_w_data", action='store_true')
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_prompt", action='store_true')
    parser.add_argument("--comp_prompts", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.)
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
        file = args.save_folder / 'ends.txt'
        f = open(file, "r")
        sentences = f.readlines()
        sentences = [s.replace('\n','') for s in sentences]
        sentences = [s.split('r=')[0] for s in sentences]
        tokenized_sentences = tokenizer(sentences, return_tensors='pt', padding=True, max_length=args.max_length)['input_ids']
        tokenized_sentences = tokenized_sentences.to(device)

    else:
        tokenized_sentences = None

    # load cvar value model
    learning_filename=args.save_folder / args.value_model
    n_quantiles = 10; hidden_dim = int(args.value_model.split('/')[0].split('_')[2])
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
    token_storage = []
    prompt_storage = []
    cvar_storage = []
    quantile_storage = []
    successes = []
    prompt_storage = []

    for i in tqdm(range(args.num_iterations)):

        # choose 3 random prompts
        if args.comp_prompts:
            prompt = ' '.join(np.random.choice(prompts, size=3))
        else:
            prompt =  np.random.choice(prompts, size=1)[0]

        #if np.random.binomial(n=1, p=0.5)==1:
        #    prompt = prompt.replace('I', 'You')

        if args.prompt is not None:
            prompt = args.prompt
            #import ipdb; ipdb.set_trace()

        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        if i==1000:
            step_by_step=True
        else:
            step_by_step=False
        #step_by_step=False

        bad_words = ["\n", "\r", '"',' "',' (',' )',' [',' ]','."','?"',',"']
        bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words] # it recomments prefix space, but not doing that right now

        # generate possible continuations
        output, other_outputs = generate(model, tokenizer,
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=inputs['input_ids'].shape[1]+args.max_length,
                            num_beams = 1,
                            temperature=args.temperature, num_return_sequences=1,
                            do_sample=True, eos_token_id=13,
                            bad_words_ids = bad_words_ids,
                            top_k = args.top_k, top_p=0.95,
                            allowed_word_ids = None,
                            data_to_restrict_w = tokenized_sentences,
                            cvar_alpha = args.alpha,
                            Z_network = Z_network,
                            tokenized_prompt = tokenized_prompt,
                            return_dict_in_generate=False,
                            step_by_step = step_by_step,
                            use_prompt_for_dist = args.use_prompt,
                            )

        decoded.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        prompt_storage.extend([prompt for _ in range(0)])

        alphas.append(other_outputs['alphas'])
        p_storage.append(other_outputs['p_storage'])
        pd_storage.append(other_outputs['pd_storage'])
        token_storage.append(other_outputs['token_storage'])
        cvar_storage.append(other_outputs['cvar_storage'])
        prompt_storage.append(prompt)
        quantile_storage.append(other_outputs['quantile_storage'])
        successes.append(other_outputs['successes'])

    assert len(decoded)==len(prompt_storage)

    # process a bit
    decoded_proc =[]
    prompt_storage2 = []
    for d, pr in zip(decoded, prompt_storage):
        include=True
        # if '\n' in d:
        #     include=False
        # if '\r' in d:
        #     include=False
        # if '[' in d or ']' in d:
        #     include=False
        if d[-1]!='.':
            include=False
        # if '?' in d:
        #     include=False
        # if has_numbers(d):
        #     include=False
        num_words = d.split(' ')
        if len(num_words)<3:
            include=False

        if d==pr:
            include=False

        if include:
            decoded_proc.append(d)
            prompt_storage2.append(pr)
        else:
            decoded_proc.append('[EXCLUDED]')
            prompt_storage2.append('[EXCLUDED]')

    decoded = decoded_proc
    prompt_storage = prompt_storage2
    assert len(decoded)==len(prompt_storage)

    ends = []
    for d,p in zip(decoded, prompt_storage):
        if d !='[EXCLUDED]':
            ends.append(d.replace(p,'').strip())
        else:
            ends.append(d)

    rewards = []
    for e in ends:
        if e !='[EXCLUDED]':
            try:
                rewards.extend(score_sentiment(e, sentiment_tokenizer, sentiment_model, device))
            except:
                rewards.extend([np.nan])
        else:
            rewards.extend([np.nan])
    assert len(rewards)==len(ends)

    decoded_unsorted = decoded.copy()
    ends_unsorted = ends.copy()
    rewards_unsorted = rewards.copy()
    sort_idx = np.argsort(rewards)
    rewards = [rewards[i] for i in sort_idx]
    decoded = [decoded[i] for i in sort_idx]
    ends = [ends[i] for i in sort_idx]

    if args.use_prompt:
        use_p = '_prompt_enc'
    else:
        use_p = ''

    if args.prompt is not None:
        prompt_save = '_'+prompt
    else:
        prompt_save = ''
    if args.seed!=2311:
        seed_save=str(args.seed)
    else:
        seed_save=''

    if args.temperature!=1.:
        temp_save = 'temp='+str(args.temperature)
    else:
        temp_save = ''

    if args.comp_prompts:
        prompt_save2 = ''
    elif args.comp_prompts==False and args.prompt==None:
        prompt_save2 = '_randprompt'


    # save results
    if args.restrict_w_data:
        filename1 = Path(args.save_folder) / f'full_generations_cvar_{args.alpha}_{args.top_k}{use_p}{prompt_save}{seed_save}{temp_save}{prompt_save2}.txt'
        filename2 = Path(args.save_folder) / f'ends_cvar_{args.alpha}_{args.top_k}{use_p}{prompt_save}{seed_save}{temp_save}{prompt_save2}.txt'
    else:
        filename1 = Path(args.save_folder) / f'full_generations_cvar_{args.alpha}_{args.top_k}{use_p}_unres{prompt_save}{seed_save}{temp_save}{prompt_save2}.txt'
        filename2 = Path(args.save_folder) / f'ends_cvar_{args.alpha}_{args.top_k}{use_p}_unres{prompt_save}{seed_save}{temp_save}{prompt_save2}.txt'

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
    other_outputs['sentences']=ends_unsorted
    other_outputs['rewards']=rewards_unsorted
    other_outputs['sentences_full']=decoded_unsorted
    other_outputs['token_storage']=token_storage
    other_outputs['prompt_storage'] = prompt_storage
    other_outputs['cvar_storage'] = cvar_storage
    other_outputs['quantile_storage']  = quantile_storage
    other_outputs['successes'] = successes

    if args.restrict_w_data:
        filename3 = Path(args.save_folder) / f'cvar_output_{args.alpha}_{args.top_k}{use_p}{prompt_save}{seed_save}{temp_save}{prompt_save2}.pkl'
    else:
        filename3 = Path(args.save_folder) / f'cvar_output_{args.alpha}_{args.top_k}{use_p}_unres{prompt_save}{seed_save}{temp_save}{prompt_save2}.pkl'

    pickle.dump(other_outputs, open(filename3, 'wb'))


if __name__ == '__main__':

    main()


    # Dataset V3
    # CUDA_VISIBLE_DEVICES=1 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 10 --value_model 'quantile_learner_101_0.1/quantile_learner_epoch95.pkl' --save_folder 'single_sentences_IYou_3' --comp_prompts

    # Random selection of 3 prompts
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --comp_prompts
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 20 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --comp_prompts
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 40 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --comp_prompts
    # CUDA_VISIBLE_DEVICES=1 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.2 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --comp_prompts

    # Alternative Slides example - 'I need to go to work early tomorrow.'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow.'
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 25 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'Tomorrow will be a busy day. I need to go to work early.'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow.' --seed 1
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.1 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow.' --seed 1

    # Slides example - 'I need to go to work early tomorrow. It will be'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.1 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.1 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 1
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.1 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.3 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2

    # Slides example - 'I need to go to work early tomorrow. It will be'
    # Lowering the temperature
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 1.0 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2 --temperature 0.8
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.1 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2 --temperature 0.8
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.3 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --prompt 'I need to go to work early tomorrow. It will be' --seed 2 --temperature 0.8

    # Random selection of 1 prompt
    # CUDA_VISIBLE_DEVICES=2 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch75.pkl' --save_folder 'single_sentences_IYou_3' --seed 2




    # Dataset V4
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch95.pkl' --save_folder 'single_sentences_IYou_4'

    # Dataset V5
    # CUDA_VISIBLE_DEVICES=1 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 10 --use_prompt --value_model 'quantile_learner_104_0.1_prompt_enc/quantile_learner_epoch95.pkl' --save_folder 'single_sentences_IYou_5'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentences_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 100 --alpha 0.05 --top_k 10 --value_model 'quantile_learner_104_0.1/quantile_learner_epoch95.pkl' --save_folder 'single_sentences_IYou_5'
