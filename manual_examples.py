
from cvar_helpers import calc_cvar_from_quantiles
from cvar_sampler import distort_probabilities
import torch
import torch.nn as nn
from transformers import (
    MinLengthLogitsProcessor,
    LogitsProcessorList,
    TopKLogitsWarper
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import rc
from matplotlib.transforms import Affine2D


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = text.get_transform() + Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height)

def get_probabilities(prompt, allowed_next_tokens, tokenizer, model, device, verbose=True):
    '''Get probabilities for the next tokens.'''

    # encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]

    # process logits (min length, set P(EOS)=0)
    logits_processor = LogitsProcessorList()
    logits_processor.append(MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id))
    next_token_scores = logits_processor(inputs['input_ids'], next_token_logits)

    # print top k
    #logits_processor2 = LogitsProcessorList()
    #logits_processor2.append(TopKLogitsWarper(top_k=5, min_tokens_to_keep=1))
    #next_token_scores2 = logits_processor2(inputs['input_ids'], next_token_scores)
    top_k=10
    next_token_probs_all = nn.functional.softmax(next_token_scores, dim=-1)
    next_token_probs_all = next_token_probs_all.detach().cpu().squeeze()
    sort_idx = torch.argsort(next_token_probs_all, dim=-1, descending=True) # can actually do by batch
    top_tokens_all = sort_idx[0:top_k]
    top_probs_all = next_token_probs_all[sort_idx][0:top_k]
    # if verbose:
    #     print(prompt)
    #     print(f'top tokens ids: {top_tokens_all}')
    #     print(f'top tokens: {tokenizer.decode(top_tokens_all)}')
    #     print(f'p:\t{top_probs_all.numpy().round(2)}')

    # Restrict to allowed tokens
    # Important note: tokens are given spaces here.
    allowed_word_ids = [tokenizer.encode(word, add_prefix_space=True) if word not in ['.',','] else tokenizer.encode(word, add_prefix_space=False) for word in allowed_next_tokens]
    #allowed_word_ids = [tokenizer.encode(word, add_prefix_space=True) for word in allowed_next_tokens]

    allowed_word_ids = [word[0] for word in allowed_word_ids] # only allow single tokens
    if allowed_word_ids is not None:
        banned_mask = torch.ones(next_token_scores.shape[1])
        banned_mask[allowed_word_ids] = 0
        banned_mask = banned_mask.bool().unsqueeze(0).to(model.device) # [1 x 50,000]
        next_token_scores = next_token_scores.masked_fill(banned_mask, -float("inf"))
    #
    # if verbose:
    #     print(f'allowed words:\t{allowed_next_tokens}')
    #     print(f'allowed words ids:\t{allowed_word_ids}')

    # soft-max (within restricted set)
    probs = nn.functional.softmax(next_token_scores, dim=-1).squeeze()
    sort_idx = torch.argsort(probs, dim=-1, descending=True) # can actually do by batch
    top_tokens = sort_idx[0:len(allowed_next_tokens)]
    top_probs = probs[sort_idx][0:len(allowed_next_tokens)]

    #import ipdb; ipdb.set_trace()

    return(top_tokens, top_probs, [tokenizer.decode(tok) for tok in top_tokens],
           top_tokens_all, tokenizer.decode(top_tokens_all), top_probs_all)


def get_distributions(prompt, token_ids, tokenizer, model, device, Z_network):
    '''Get valence distributions for the next tokens.'''

    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)['input_ids']
    n_quantiles = Z_network.num_quant
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    Vp = []
    Vp_quantiles = []
    for tok_id in token_ids:
        tok_id = tok_id.unsqueeze(-1).to(model.device)
        prompt_w_tok = torch.cat([tokenized_prompt, tok_id[:, None]], dim=-1)

        with torch.no_grad():
            outputs = model(input_ids=prompt_w_tok,
                          attention_mask=torch.ones_like(prompt_w_tok),
                          return_dict=True,
                          output_hidden_states=True)
            states = outputs['hidden_states'][-1][:,-1,:].unsqueeze(1)
            thetas = Z_network(states).detach().cpu().numpy().squeeze()
        cvars = calc_cvar_from_quantiles(thetas, taus, alphas)
        Vp.append(cvars)
        Vp_quantiles.append(thetas)

    Vp = np.array(Vp)
    Vp_quantiles = np.array(Vp_quantiles)

    return(Vp, Vp_quantiles)

def get_prompt_distribution(prompt, tokenizer, model, device, Z_network):

    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)['input_ids']
    n_quantiles = Z_network.num_quant
    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    with torch.no_grad():
        outputs = model(input_ids=tokenized_prompt,
                      attention_mask=torch.ones_like(tokenized_prompt),
                      return_dict=True,
                      output_hidden_states=True)
        states = outputs['hidden_states'][-1][:,-1,:].unsqueeze(1)
        thetas = Z_network(states).detach().cpu().numpy().squeeze()
    cvars = calc_cvar_from_quantiles(thetas, taus, alphas)

    return(thetas, cvars)


def plot_distributions(tokens, Vp_quantiles, probs, probs_distorted,
                        vert=False, kde=True, xlim=[-1,1], fs_title=16, fs_probs=16,
                        include_distorted_p=True, include_quantiles=True, include_distribution=True,
                        alpha=None, taus=None, xis=None, fig=None, axes=None, next_token=None, next_alpha=None):

    if fig is None:
        if vert:
            fig, axes = plt.subplots(len(tokens),1, figsize=(3,2.6*len(tokens)), dpi=100)
        else:
            fig, axes = plt.subplots(1,len(tokens), figsize=(3*len(tokens),3), dpi=100)

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
                plt.axvline(x=q, linestyle='--', linewidth=1.1, alpha=plot_alpha) #color=color)

        #sns.kdeplot(quantiles, bw_adjust=.5, fill=True)
        density = gaussian_kde(quantiles)
        spacing = 0.1
        x = np.arange(xlim[0],xlim[1],spacing)
        density_est = density(x)
        #import ipdb; ipdb.set_trace()
        # density_est = np.zeros_like(x)
        # for i, xi in enumerate(x):
        #     density_est[i]=np.sum(quantiles<x)
        lw = 0.1
        if include_distribution:
            bars = plt.bar(x, height=density_est, width=spacing, alpha=1., edgecolor='k',linewidth=lw)

            if alpha is not None and include_distorted_p:
                assert len(bars)==len(x)
                closest_tau = taus[np.argmin(np.abs(alpha*xis[i]-taus))]
                closest_tau_i = list(taus).index(closest_tau)
                for b, xi in zip(bars,x):
                    if xi>=np.round(quantiles[closest_tau_i],2):
                        b.set_alpha(0.3)
                    if alpha*xis[i]<0.01:
                        b.set_alpha(0.3)


        ylim = plt.gca().get_ylim()

        sns.despine(left=True)
        plt.xlim(xlim)
        title = f'"{tokens[i].strip()}"'
        if (next_token==tokens[i]) and include_distorted_p:
            plt.title(title, fontsize=fs_title, pad=10, fontweight='bold')
            if next_alpha is not None:
                plt.text(0.5,0.5,r'$\alpha$='+str(np.round(next_alpha,2)), fontsize=fs_probs-4)
        else:
            plt.title(title, fontsize=fs_title, pad=10)

        text = f'\np={probs[i]:.2f}'
        xdist_left=1.1
        vert_spacing = 0.075
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
