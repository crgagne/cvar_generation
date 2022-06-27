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
import pickle

from generator import generate
from rl_learner import TD_Learner
from cvar_helpers import calc_cvar_from_quantiles
from cvar_sampler import distort_probabilities
from train_rl_batch_sentence_chains import average_states_by_period

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
        if d[0]=='.':
            include=False

        if d[-1]!='.':
            include=False
        #if '?' in d:
        #    include=False
        if has_numbers(d):
            include=False
        num_words = d.split(' ')
        if len(num_words)<3:
            include=False
        if d.count('.')!=n_periods:
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
    parser.add_argument("--value_model", type=str, default="quantile_learner_100_0.1/quantile_learner_epoch48.pkl")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--save_folder", type=str, default='sentence_chains_I_4')
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cvar_type", type=str, default='pCVaR')
    parser.add_argument("--bias_by_reward", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)

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

    # load cvar value model
    learning_filename=args.save_folder / args.value_model
    n_quantiles = 10; hidden_dim = int(args.value_model.split('/')[0].split('_')[2])
    Z_network = TD_Learner(config.n_embd, n_quantiles, hidden_dim=hidden_dim).to(device)
    Z_network.load_state_dict(torch.load(learning_filename))
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    outer_storage = {'alpha_storage':[],
               'p_storage':[],
               'pd_storage':[],
               'state_storage':[],
               'cvar_storage':[],
               'quantile_storage':[],
               'rewards_storage':[],
               'sentences_storage':[],
                }

    bad_words = ['."', "\n", "\r", '"']
    bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
    bad_words_ids.append([526])

    for i in tqdm(range(args.num_iterations)):


        inner_storage = {'alpha_storage':[],
                   'p_storage':[],
                   'pd_storage':[],
                   'state_storage':[],
                   'cvar_storage':[],
                   'quantile_storage':[],
                   'rewards_storage':[],
                   'sentences_storage':[],
                    }

        s0= ['Let me think about what I need to do tomorrow, because it will be a busy day.',
            'I need to get the kids ready for school.' ,
            'I need to feed the dog.']
        #s0= ['Tomorrow I will wake up early.',  'I will leave the house.']

        s0=['Tomorrow is going to be a busy day.', 'I need to go the doctors. What if'] # this actually looks pretty good...
        # # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl" --verbose
        # with 20 examples
        s0=['Tomorrow is going to be a busy day.', 'I need to take that test. What if']
        s0=['Tomorrow is going to be a busy day.', 'I need to take that test. What if I fail? Then what?']
        s0=['Tomorrow is going to be a busy day.', 'I need to go to the grocery store. What if']

        # selecting s0 and s2
        s0s = ['Tomorrow will be an easy day.','Tomorrow will be a busy day.', 'Tomorrow will be a difficult day.']
        s1s = ["I need to go to the doctor's.", 'I need to go to work.', 'I need to go to the grocery store.', 'I need to clean the house.',  'I need to take that test.',
                "I will attend my friend's party.", "I will go to the gym.", "I will go on a bike ride.", "I might need to work late." ]
        s0 = [np.random.choice(s0s), np.random.choice(s1s)]

        # What if I fail? Then I'll never be able to get a good job.
        # What if I fail? Then I'll just study hard and do better next time.

        s = ' '.join(s0)

        if args.prompt is not None:
            s = args.prompt

        state_storage = s0
        rewards_received = []
        probs_selected = []
        cvar_alpha = args.alpha
        inner_storage['alpha_storage'].append(cvar_alpha)

        for step in range(args.steps):

            # tokenize
            inputs = tokenizer(s, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            mask = inputs['attention_mask']

            # generate possible continuations
            with torch.no_grad():
                output, other_outputs = generate(model, tokenizer,
                                    input_ids=input_ids,
                                    attention_mask=mask,
                                    max_length=input_ids.shape[1]+args.max_length,
                                    num_beams = 1, # not implemented
                                    temperature=0.8, # was 0.9 before
                                    num_return_sequences=14,
                                    do_sample=True,
                                    eos_token_id=13,
                                    bad_words_ids = bad_words_ids,
                                    top_k = 50, top_p=0.90, # was k=0 before with longer prompt; top_p=0.95 before
                                    allowed_word_ids = None,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    )

            sel_logits = np.array(other_outputs['selected_tok_scores_all']).shape
            gen_sequences = output.sequences[:, input_ids.shape[-1]:]
            seq_probs = torch.stack(output.scores, dim=1).softmax(-1)  # -> shape [batch, seq_size, vocab_size]
            gen_probs = torch.gather(seq_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            mean_prob_per_sequence = np.mean(gen_probs.detach().cpu().numpy()[:,:], axis=1)

            decoded = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
            decoded, include_list = process_decoded(decoded, n_periods=s.count('.')+1)
            include_list = np.array(include_list)

            mean_prob_per_sequence = mean_prob_per_sequence[include_list==True]

            if len(decoded)==0:
                break

            # modify candidates, remove ones that are too similar
            sp_candidates = []
            include_list = []
            for d in decoded:
                include=True
                candidate = d.split('.')[-2].strip()+'.'
                assert candidate.count('.')==1

                seq_matches = np.array([SequenceMatcher(None, candidate, s_prev).ratio() for s_prev in state_storage])
                if np.any(seq_matches>0.8):
                    include=False
                if candidate=='.':
                    include=False

                include_list.append(include)
                if include:
                    sp_candidates.append(candidate)

            if len(sp_candidates)==0:
                break

            mean_prob_per_sequence = mean_prob_per_sequence[np.array(include_list)==True]

            # calculate probabilities
            res = entailment_classifier(s, sp_candidates)
            probs = res['scores']
            probs = probs/np.sum(probs)

            # re order
            reorder_idx = [sp_candidates.index(s) for s in res['labels']]
            sp_candidates_re = [sp_candidates[i] for i in reorder_idx]
            assert sp_candidates_re==res['labels']
            sp_candidates = sp_candidates_re

            inner_storage['p_storage'].append(probs)

            mean_prob_per_sequence = mean_prob_per_sequence[reorder_idx]
            p_mean = mean_prob_per_sequence / np.sum(mean_prob_per_sequence)

            # calculate intermediate rewards
            rewards = score_sentiment(sp_candidates, sentiment_tokenizer, sentiment_model, device)

            # get next state distribution, quantiles/cvar
            Vp = []
            Vp_quantiles = []
            for _sp in sp_candidates:

                _s = s + ' ' + _sp # form the full chain (temporarily)
                input = tokenizer(_s, return_tensors='pt').to(device)
                input_ids = input['input_ids']
                mask = input['attention_mask']
                with torch.no_grad():
                    output = model(input_ids=input_ids, attention_mask=mask,output_hidden_states=True)
                    states = output['hidden_states'][-1]
                    n_periods = _s.count('.')
                    try:
                        states, _ = average_states_by_period(states, mask.unsqueeze(-1), input_ids, device,
                                                                n_periods=n_periods, period_tok_id=13, pad_tok_id=50256)
                    except:
                        print(_sp)
                        import ipdb; ipdb.set_trace()
                    # states will be (batch_size, num_sentences, hiddden_state_dim)
                    theta_hats = Z_network(states).detach().cpu().numpy()#.round(2) # calculates it for each state (batch_size, num_sentences, n_quantiles)
                    thetas = theta_hats[:,-1,:].squeeze() # take the theta hats for the candidate representation
                    # check sh
                cvars = calc_cvar_from_quantiles(thetas, taus, alphas)
                Vp.append(cvars)
                Vp_quantiles.append(thetas)

            Vp = np.array(Vp)
            Vp_quantiles = np.array(Vp_quantiles) # just for visualizing

            if args.bias_by_reward:
                for rr,r in enumerate(rewards):
                    Vp[rr,:]=r # set all CVaRs to the single reward of the next sentence candidate

            inner_storage['quantile_storage'].append(Vp_quantiles)
            inner_storage['cvar_storage'].append(Vp)

            # distort probabilites
            probs_distorted, xis, extra = distort_probabilities(probs, cvar_alpha, alphas, Vp,
                                                                max_inner_iters=10, multi_starts_N=5, same_answer_ns=2, same_answer_tol=1e-3)
            probs_distorted = probs_distorted / np.sum(probs_distorted)
            inner_storage['pd_storage'].append(probs_distorted)

            # select next state
            sp = np.random.choice(sp_candidates, p=probs_distorted)
            selected_prob = probs[sp_candidates.index(sp)] # real probs
            probs_selected.append(selected_prob)

            # adjust alpha
            if args.cvar_type=='pCVaR':
                cvar_alpha = float(xis[sp_candidates.index(sp)]*cvar_alpha)

            cvar_alpha = np.max(np.min((cvar_alpha,1)),0)
            inner_storage['alpha_storage'].append(cvar_alpha)

            # store rewards
            selected_reward = rewards[sp_candidates.index(sp)]
            rewards_received.append(selected_reward)
            state_storage.append(sp)
            inner_storage['rewards_storage'].append(selected_reward)

            if args.verbose:
                print(f'{s}')
                for _sp, _p, _pd, _r, _vp, _pm in zip(sp_candidates, probs, probs_distorted, rewards, Vp_quantiles, p_mean):
                    print(f' .. {_sp} p={_p:.2f}({_pm:0.2f}) pd={_pd:.2f} true_r={_r:.2f} q={np.round(_vp,1)[[0,5,9]]}')

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

        inner_storage['sentences_storage'].append(traj)

        for key in inner_storage.keys():
            outer_storage[key].append(inner_storage[key])

        if 'nt_rewards' in args.value_model:
            nt_rewards = '_nt'
        else:
            nt_rewards = ''

        if args.cvar_type!='pCVaR':
            cvar_type = '_fCVaR'
        else:
            cvar_type = ''
        if args.bias_by_reward:
            cvar_type +='_rbias'

        # save sentence chains
        filename1 = Path(args.save_folder) / f'generations_alpha{args.alpha}_steps{args.steps}{nt_rewards}{cvar_type}.txt'
        with open(filename1, 'w' if i==0 else 'a') as f1:
            f1.write(traj)

    # save the rest of the results
    filename2 = Path(args.save_folder) / f'cvar_output_alpha{args.alpha}_steps{args.steps}{nt_rewards}{cvar_type}.pkl'
    pickle.dump(outer_storage, open(filename2, 'wb'))

if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=1 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_3' --value_model "quantile_learner_100_0.1/quantile_learner_epoch60.pkl"
    # CUDA_VISIBLE_DEVICES=1 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.1 --save_folder 'sentence_chains_I_3' --value_model "quantile_learner_100_0.1/quantile_learner_epoch60.pkl"
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_2' --value_model "quantile_learner_100_0.1/quantile_learner_epoch60.pkl"
    # CUDA_VISIBLE_DEVICES=1 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.08 --save_folder 'sentence_chains_I_3' --value_model "quantile_learner_100_0.1/quantile_learner_epoch60.pkl"

    # V4
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1/quantile_learner_epoch48.pkl" --verbose
    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl"
    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl" --steps 6 --verbose
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl" --steps 6 --cvar_type 'fCVaR'
    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl" --steps 3 --cvar_type 'fCVaR' --bias_by_reward

    # CUDA_VISIBLE_DEVICES=3 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_4' --value_model "quantile_learner_100_0.1_nt_rewards/quantile_learner_epoch48.pkl" --verbose

    # V5
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_100_0.1_composite_rewards_slow_lr/quantile_learner_epoch19.pkl"
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_10_0.1_composite_rewards_slow_lr/quantile_learner_epoch19.pkl"

    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 200 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_10_0.1_composite_rewards/quantile_learner_epoch48.pkl"



    # with prompt
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_100_0.1_composite_rewards/quantile_learner_epoch48.pkl" --verbose --prompt "Tomorrow will be a busy day. I need to go to the doctor's."
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_10_0.1_composite_rewards/quantile_learner_epoch48.pkl" --verbose --prompt "Tomorrow will be a busy day. I need to go to the doctor's."
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_10_0.1_composite_rewards/quantile_learner_epoch48.pkl" --prompt "Tomorrow will be a busy day. I need to go to the doctor's."
    # CUDA_VISIBLE_DEVICES=2 python generate_sentence_chains_w_cvar.py  --model models/pretrained/gpt2-large --num_iterations 50 --alpha 0.05 --save_folder 'sentence_chains_I_5' --value_model "quantile_learner_10_0.1_composite_rewards/quantile_learner_epoch48.pkl" --prompt "Tomorrow will be a busy day. I need to go to the doctor's."
