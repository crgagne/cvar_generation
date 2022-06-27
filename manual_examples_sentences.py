from train_rl_batch_sentence_chains import average_states_by_period
from cvar_helpers import calc_cvar_from_quantiles
from cvar_sampler import distort_probabilities
import torch
import torch.nn as nn
from transformers import (
    MinLengthLogitsProcessor,
    LogitsProcessorList,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from manual_examples import rainbow_text


def get_probabilities(chain, allowed_next_sentences, entailment_classifier, verbose=True):
    '''Get probabilities for the next sentences.'''
    res = entailment_classifier(chain, allowed_next_sentences)
    probs = res['scores']
    probs = probs/np.sum(probs)
    return(probs)

def get_distributions(chain, allowed_next_sentences, tokenizer, model, device, Z_network):
    '''Get valence distributions for the next sentences.'''

    n_quantiles = Z_network.num_quant
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    Vp = []
    Vp_quantiles = []
    for next_sentence in allowed_next_sentences:
        if next_sentence[0]!=' ':
            next_sentence = ' '+next_sentence
        chain_w_next = chain + next_sentence
        input_ids = tokenizer(chain_w_next, return_tensors='pt').to(device)['input_ids']
        mask = tokenizer(chain_w_next, return_tensors='pt').to(device)['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                          attention_mask=mask,
                          return_dict=True,
                          output_hidden_states=True)
            states = outputs['hidden_states'][-1]
            n_periods = chain_w_next.count('.')
            states, _ = average_states_by_period(states, mask.unsqueeze(-1), input_ids, device,
                                                 n_periods=n_periods, period_tok_id=13, pad_tok_id=50256)
            thetas = Z_network(states).detach().cpu().numpy()
            thetas = thetas[:,-1,:].squeeze() # take the theta hats for last hidden state (last sentence)
        cvars = calc_cvar_from_quantiles(thetas, taus, alphas)
        Vp.append(cvars)
        Vp_quantiles.append(thetas)

    Vp = np.array(Vp)
    Vp_quantiles = np.array(Vp_quantiles)
    return(Vp, Vp_quantiles)

def get_chain_distribution(chain, tokenizer, model, device, Z_network):

    input_ids = tokenizer(chain, return_tensors='pt').to(device)['input_ids']
    mask = tokenizer(chain, return_tensors='pt').to(device)['attention_mask']

    n_quantiles = Z_network.num_quant
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                      attention_mask=torch.ones_like(input_ids),
                      return_dict=True,
                      output_hidden_states=True)
        states = outputs['hidden_states'][-1]
        n_periods = chain.count('.')
        states, _ = average_states_by_period(states, mask.unsqueeze(-1), input_ids, device,
                                             n_periods=n_periods, period_tok_id=13, pad_tok_id=50256)
        thetas = Z_network(states).detach().cpu().numpy().squeeze()
    #cvars = calc_cvar_from_quantiles(thetas, taus, alphas) # can't do unless apply to each hidden state seperately.

    return(thetas, None)

def plot_distributions(allowed_next, Vp_quantiles, probs, probs_distorted, vert=False,
                       alpha=None, taus=None, xis=None, xlim=[-3,3],
                       fig=None, axes=None, next_sent=None,
                       fs_title=16, fs_probs=16,
                       include_distorted_p=True, include_quantiles=True, include_distribution=False, next_alpha=None):

    if fig is None:
        if vert:
            fig, axes = plt.subplots(len(allowed_next),1, figsize=(3,3*len(allowed_next)))
        else:
            fig, axes = plt.subplots(1,len(allowed_next), figsize=(3*len(allowed_next),3))

        axes = axes.flatten()

    for i in range(Vp_quantiles.shape[0]):
        quantiles = Vp_quantiles[i, :]
        plt.sca(axes[i])
        if include_quantiles:
            for q, tau in zip(quantiles, taus):
                if alpha is not None and include_distorted_p:
                    closest_tau = taus[np.argmin(np.abs(alpha*xis[i]-taus))]
                    closest_tau_i = list(taus).index(closest_tau)
                    if tau<=closest_tau:
                        color='r'
                        plot_alpha = 1.0
                    else:
                        color='b'
                        plot_alpha = 0.3
                    if alpha*xis[i]<0.01:
                        color='b'
                        plot_alpha = 0.3
                else:
                    plot_alpha=1.
                plt.axvline(x=q, linestyle='--', linewidth=1.1, alpha=plot_alpha)


        lw = 0.1
        if include_distribution:
            bars = plt.bar(x, height=density_est, width=spacing, alpha=1., edgecolor='k',linewidth=lw)

            if alpha is not None:
                assert len(bars)==len(x)
                closest_tau = taus[np.argmin(np.abs(alpha*xis[i]-taus))]
                closest_tau_i = list(taus).index(closest_tau)
                for b, xi in zip(bars,x):
                    if xi>=np.round(quantiles[closest_tau_i],2):
                        b.set_alpha(0.3)
                    if alpha*xis[i]<0.01:
                        b.set_alpha(0.3)

        sns.despine(left=True)

        plt.xlim(xlim)
        title = f'"{allowed_next[i].strip()}"'
        if next_sent==allowed_next[i] and include_distorted_p:
            plt.title(title, fontsize=fs_title, pad=10, fontweight='bold')
            if next_alpha is not None:
                plt.text(0.5,0.5,r'$\alpha$='+str(np.round(next_alpha,2)), fontsize=fs_probs-2)
        else:
            plt.title(title, fontsize=fs_title, pad=10)

        text = f'\np={probs[i]:.2f}'
        xdist_left=1.1
        vert_spacing = 0.075
        ylim = plt.gca().get_ylim()
        rainbow_text(xlim[0]-xdist_left, (0.5+vert_spacing)*ylim[1], [text], ['black',], size=fs_probs)

        if include_distorted_p:
            text = f'\np={probs_distorted[i]:.2f}'
            if probs_distorted[i]>probs[i]:
                color='red'
            else:
                color='green'
            rainbow_text(xlim[0]-xdist_left, (0.5-vert_spacing)*ylim[1], [text], [color], size=fs_probs)

        plt.yticks([])
        plt.tick_params(right = False , labelleft = False)
        if include_distribution==False and include_quantiles==False:
            plt.axis('off')
        #plt.axvline(x=0,linestyle='-',color='k')

    plt.tight_layout()
