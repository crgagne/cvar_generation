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
import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm
import scipy
import re

from generator import generate

from difflib import SequenceMatcher


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def process_decoded(decoded, n_periods):

    # process a bit
    decoded_proc =[]
    include_list = []
    for d in decoded:
        include=True
        d = d.replace('\n','').replace('\r','').replace('"','')
        d = d.replace('[','').replace(']','')

        if (d[-1]!='.') and (d[-1]!='?'):
            include=False

        if has_numbers(d):
            include=False
        num_words = d.split(' ')
        if len(num_words)<3:
            include=False
        if (d.count('.')+d.count('?'))!=n_periods:
            include=False

        include_list.append(include)
        if include:
            decoded_proc.append(d)

    return(decoded_proc, include_list)

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
    parser.add_argument("--save_folder", type=str, default='sentence_chains_I_5')
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--steps", type=int, default=3)
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

    # reward model
    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    bad_words = ['."','?"', "\n", "\r", '"', '_', '"', '\n', 'said','.,','?,','.?']
    bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
    bad_words_ids.extend([tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words])
    bad_words_ids = [bw for bw in bad_words_ids if len(bw)==1]

    for i in tqdm(range(args.num_iterations)):

        #s = 'I woke up early. Then I brushed my teeth.'

        s0= ['Tomorrow I will wake up early.',  'I will leave the house.']
        #s0= ['Tomorrow I will wake up early as I always do.',  'I will leave the house to do errands.']
        #s0= ['Tomorrow I will wake up early as I always do.',  'I will brush my teeth, eat breakfast, and leave the house to do errands.']
        s0= ['Let me think about what I need to do tomorrow, because it will be a busy day.',
            'I need to get the kids ready for school.' ,
            'I need to feed the dog.']
        s0= ['Tomorrow will be a busy day.',
            'I need to go to the doctors.']


        # selecting s0 and s2
        s0s = ['Tomorrow will be an easy day.','Tomorrow will be a busy day.', 'Tomorrow will be a difficult day.']
        s1s = ["I need to go to the doctor's.", 'I need to go to work.', 'I need to go to the grocery store.', 'I need to clean the house.',  'I need to take that test.',
                "I will attend my friend's party.", "I will go to the gym.", "I will go on a bike ride.", "I might need to work late." ]

        s0 = [np.random.choice(s0s), np.random.choice(s1s)]

        s = ' '.join(s0)

        rewards_received = []
        probs_selected = []
        state_storage = s0

        for step in range(args.steps):

            # add partial stem
            #if step==0:
            tmp = np.random.choice([0,1,2],p=[0.25,0.25,0.5])
            if tmp==0:
                s_tmp = s + ' What if'
            elif tmp==1:
                s_tmp = s + ' Then'
            elif tmp==2:
                s_tmp = s

            # tokenize
            inputs = tokenizer(s_tmp, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            mask = inputs['attention_mask']

            # generate possible continuations
            with torch.no_grad():
                output, other_outputs = generate(model, tokenizer,
                                    input_ids=input_ids,
                                    attention_mask=mask,
                                    max_length=input_ids.shape[1]+args.max_length,
                                    num_beams = 1,
                                    temperature=0.8, # was 0.9 with longer prompt
                                    num_return_sequences=14,
                                    do_sample=True,
                                    eos_token_id=13,
                                    eos_token_id2=30,
                                    bad_words_ids = bad_words_ids,
                                    top_k = 50, top_p=0.90, # was k=0 before with longer prompt; top_p=0.95 before
                                    allowed_word_ids = None,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    )

            # https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
            sel_logits = np.array(other_outputs['selected_tok_scores_all']).shape
            gen_sequences = output.sequences[:, input_ids.shape[-1]:]
            seq_probs = torch.stack(output.scores, dim=1).softmax(-1)  # -> shape [batch, seq_size, vocab_size]
            gen_probs = torch.gather(seq_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            gmean_prob_per_sequence = scipy.stats.mstats.gmean(gen_probs.detach().cpu().numpy()[:,:], axis=1)
            mean_prob_per_sequence = np.mean(gen_probs.detach().cpu().numpy()[:,:], axis=1)
            unique_prob_per_sequence = gen_probs.prod(-1)

            seq_logits = torch.stack(output.scores, dim=1)
            gen_logits = torch.gather(seq_logits, 2, gen_sequences[:, :, None]).squeeze(-1)
            gen_logits[gen_logits==-torch.inf]=0.
            gen_logits_per_sequence = gen_logits.sum(-1).detach().cpu().numpy()

            # losses = []
            # with torch.no_grad():
            #     for i in range(output.sequences.shape[0]):
            #         losses.append(float(model(input_ids = output.sequences[i], labels=output.sequences[i])['loss'].detach().cpu()))

            decoded = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
            decoded, include_list = process_decoded(decoded, n_periods=(s.count('.')+s.count('?')+1))
            include_list = np.array(include_list)

            gmean_prob_per_sequence = gmean_prob_per_sequence[include_list==True]
            mean_prob_per_sequence = mean_prob_per_sequence[include_list==True]
            gen_logits_per_sequence = gen_logits_per_sequence[include_list==True]
            #losses = np.array(losses)[include_list==True]

            if len(decoded)==0:
                break

            # modify candidates, remove ones that are too similar
            sp_candidates = []
            include_list = []
            for d in decoded:
                include=True
                candidate = d.replace(s, '').strip()
                if (candidate.count('.')+candidate.count('?'))!=1:
                    include=False

                seq_matches = np.array([SequenceMatcher(None, candidate, s_prev).ratio() for s_prev in state_storage])
                if np.any(seq_matches>0.8):
                    include=False

                include_list.append(include)
                if include:
                    sp_candidates.append(candidate)

            if len(sp_candidates)==0:
                break

            include_list = np.array(include_list)
            gmean_prob_per_sequence = gmean_prob_per_sequence[include_list==True]
            mean_prob_per_sequence = mean_prob_per_sequence[include_list==True]
            gen_logits_per_sequence = gen_logits_per_sequence[include_list==True]

            # calculate probabilities
            res = entailment_classifier(s, sp_candidates)
            probs = np.array(res['scores'])
            probs = np.exp(probs/1.05) / np.sum(np.exp(probs/1.05)) # slightly extra noise to prevent repetition
            probs = probs/np.sum(probs)

            # re order
            reorder_idx = [sp_candidates.index(s) for s in res['labels']]
            sp_candidates_re = [sp_candidates[i] for i in reorder_idx]
            assert sp_candidates_re==res['labels']
            sp_candidates = sp_candidates_re

            gmean_prob_per_sequence = gmean_prob_per_sequence[reorder_idx]
            mean_prob_per_sequence = mean_prob_per_sequence[reorder_idx]
            gen_logits_per_sequence = gen_logits_per_sequence[reorder_idx]
            #losses = losses[reorder_idx]

            #p_gmean = gmean_prob_per_sequence / np.sum(gmean_prob_per_sequence)
            p_mean = mean_prob_per_sequence / np.sum(mean_prob_per_sequence)
            p_log = gen_logits_per_sequence / np.sum(gen_logits_per_sequence)
            #import ipdb; ipdb.set_trace()
            #p_loss = -1*losses / np.sum(-1*losses)

            sp = np.random.choice(sp_candidates, p=probs)
            #sp = np.random.choice(sp_candidates, p=p_mean)
            selected_prob = probs[sp_candidates.index(sp)]
            probs_selected.append(selected_prob)

            # calculate intermediate rewards
            rewards = score_sentiment(sp_candidates, sentiment_tokenizer, sentiment_model, device)
            selected_reward = rewards[sp_candidates.index(sp)]
            rewards_received.append(selected_reward)
            state_storage.append(sp)

            if args.verbose:
                print(f'{s}')
                #for _sp,_p,_r in zip(sp_candidates, probs, rewards):
                #    print(f' .. {_sp} p={_p:.3f} r={_r:.3f}') # v..
                for _sp,_p,_pm,_pl, _r in zip(sp_candidates, probs, p_mean, p_log, rewards):
                    print(f' .. {_sp} p={_p:.3f} pm={_pm:.3f} pl={_pl:.3f} r={_r:.3f}') # v..

            s =  s + ' ' +sp

            if args.verbose:
                print(f'{s} p={selected_prob:.3f} r={selected_reward:.3f}')
                import ipdb; ipdb.set_trace()

        # can also score sentiment of the entire thing here
        Ret = score_sentiment(s, sentiment_tokenizer, sentiment_model, device)[0]
        Ret = np.round(Ret,3)

        probs_selected_str = ','.join([str(np.round(p,3)) for p in probs_selected])
        rewards_received_str = ','.join([str(np.round(p,3)) for p in rewards_received])
        traj = f'{s} p={probs_selected_str} r={rewards_received_str} r_all={Ret}\n'

        # save results
        filename1 = Path(args.save_folder) / f'generations_seed{args.seed}_whatif.txt'
        with open(filename1, 'w' if i==0 else 'a') as f1:
            f1.write(traj)


if __name__ == '__main__':

    main()


    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_whatif.py  --model models/pretrained/gpt2-large --num_iterations 10000 --seed 2 --verbose
