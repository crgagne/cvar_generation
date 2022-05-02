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
    pipeline
)
from transformers import pipeline
from models import GPT2CustomDoubleHeadsModel
import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm

from generator import generate


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def process_decoded(decoded):

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

    return(decoded_proc)

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
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--save_folder", type=str, default='sentence_chains_I_1')
    #parser.add_argument("--prompt_list", type=str, default="prompt_list.txt")
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--verbose", action='store_true')
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

    # entailment model (although using it in zero-shot mode. more like semantic similarity)
    entailment_classifier = pipeline("zero-shot-classification", model='models/pretrained/nli-distilroberta-base')

    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);


    for i in tqdm(range(args.num_iterations)):

        s = 'I woke up early. I brushed my teeth.'
        #s = 'Tomorrow I will wake up early. I will brush my teeth. I will have breakfast.'
        #s = 'Tomorrow I will wake up early. I will have breakfast. I will leave the house.'
        #s = 'Tomorrow I will wake up early and have my usual breakfast. Afterwards, I will take the train to work.'
        rewards_received = []
        probs_selected = []

        try:

            for step in range(3):

                # tokenize
                inputs = tokenizer(s, return_tensors='pt').to(device)

                # generate possible continuations
                with torch.no_grad():
                    output = generate(model, tokenizer,
                                        input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        max_length=inputs['input_ids'].shape[1]+args.max_length, num_beams = 1,
                                        temperature=1, num_return_sequences=10,
                                        do_sample=True, eos_token_id=13,
                                        bad_words_ids = None,
                                        top_k = 0, top_p=0.95,
                                        allowed_word_ids = None,
                                        output_scores=True,
                                        return_dict_in_generate=True,
                                        )

                decoded = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
                decoded = process_decoded(decoded)

                if len(decoded)==0:
                    break

                scores = output['scores']

                # calculate probabilities
                sp_candidates = []
                for d in decoded:
                    sp_candidates.append(d.split('.')[-2].strip()+'.')
                res = entailment_classifier(s, sp_candidates)
                probs = res['scores']
                probs = probs/np.sum(probs)
                sp = np.random.choice(sp_candidates, p=probs)
                selected_prob = probs[sp_candidates.index(sp)]
                probs_selected.append(selected_prob)

                # calculate intermediate rewards
                rewards = score_sentiment(sp_candidates, sentiment_tokenizer, sentiment_model, device)
                selected_reward = rewards[sp_candidates.index(sp)]
                rewards_received.append(selected_reward)

                if args.verbose:
                    print(f'{s}')
                    for _sp,_p,_r in zip(sp_candidates, probs, rewards):
                        print(f' .. {_sp} p={_p:.3f} r={_r:.3f}') # v..

                s =  s + ' ' +sp

                if args.verbose:
                    print(f'{s} p={selected_prob:.3f} r={selected_reward:.3f}')
                    import ipdb; ipdb.set_trace()

            probs_selected_str = ','.join([str(np.round(p,3)) for p in probs_selected])
            rewards_received_str = ','.join([str(np.round(p,3)) for p in rewards_received])
            traj = f'{s} p={probs_selected_str} r={rewards_received_str}\n'

            # save results
            filename1 = Path(args.save_folder) / 'generations.txt'
            with open(filename1, 'w' if i==0 else 'a') as f1:
                f1.write(traj)

        except:
            import ipdb; ipdb.set_trace()

if __name__ == '__main__':

    main()


    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains.py  --model models/pretrained/gpt2-large --num_iterations 1000
    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains.py  --model models/pretrained/gpt2-large --num_iterations 100
