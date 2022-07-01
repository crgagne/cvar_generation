from transformers import (
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        StoppingCriteriaList,
        NoBadWordsLogitsProcessor)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.generation_utils import SampleDecoderOnlyOutput

import torch
import torch.nn as nn
import numpy as np

from cvar_helpers import calc_cvar_from_quantiles
from cvar_sampler import distort_probabilities

def expand_inputs_for_generation(
    input_ids: torch.LongTensor,
    expand_size: int = 1,
    attention_mask: torch.LongTensor = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:

    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    return input_ids, model_kwargs

def get_logits_warper(model, top_k: int = None, top_p: float = None, temperature: float = None, num_beams: int = None) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
    """
    # init warp parameters
    top_k = top_k if top_k is not None else model.config.top_k
    top_p = top_p if top_p is not None else model.config.top_p
    temperature = temperature if temperature is not None else model.config.temperature

    # instantiate warpers list
    warpers = LogitsProcessorList()

    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        #print('warning: changed to doing top_k after top_p')
    return warpers

def sample(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    eos_token_id2: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    allowed_word_ids = None,
    data_to_restrict_w = None,
    cvar_alpha = 1.0,
    Z_network = None,
    tokenized_prompt = None,
    step_by_step=False,
    use_prompt_for_dist = False,
    flip_rewards = False,
    pcvar = True,
    verbose = True,
    **model_kwargs,
) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:

    if step_by_step:
        verbose=True

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    # CVaR related set-up
    if Z_network is not None:
        n_quantiles = Z_network.num_quant
        taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
        alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one

    if cvar_alpha!=1.:
        sample_w_cvar=True
    else:
        sample_w_cvar=False

    alpha_storage = [cvar_alpha]
    p_storage = []
    pd_storage = []
    token_storage = []
    cvar_storage = []
    quantile_storage = []
    successes = []
    selected_tok_scores_all = []

    # auto-regressive generation
    while True:

        # if generating with a prompt, but don't want to use the prompt for getting next state probabilities.
        if tokenized_prompt is not None:
            input_ids_wo_prompt = input_ids.detach().clone()
            input_ids_wo_prompt = input_ids_wo_prompt[:, tokenized_prompt.shape[1]:]

        # prepare model inputs this will be subclasased by the model
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        with torch.no_grad(): # I'm adding in

            #forward pass to get next token (more efficient)
            if tokenized_prompt is None:
                outputs = model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            else: # (less efficient, but necessary when calling the model below for CVaR; not sure why atm)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=model_inputs['attention_mask'],
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

        try:
            next_token_logits = outputs.logits[:, -1, :] #  shape is [num_seq, seq_len, vocab_size]
        except:
            next_token_logits = outputs[0][:, -1, :]

        # pre-process distribution
        #   at the moment, it's just a minimum length processor, and if the sequence is not long enough,
        #   it sets the logit for EOS to -inf, so that it can't be selected
        next_token_scores = logits_processor(input_ids, next_token_logits) # logits-->'scores';

        # manually restrict using a known dataset
        #   (taking into account the prompt)
        if data_to_restrict_w is not None:
            if input_ids_wo_prompt.shape[1]==0:
                allowed_next_toks = list(data_to_restrict_w[:,0].detach().cpu().numpy())
            else:
                sent_sel  = (data_to_restrict_w[:,0:input_ids_wo_prompt.shape[1]]==input_ids_wo_prompt).all(axis=1)
                allowed_next_toks = list(data_to_restrict_w[sent_sel,input_ids_wo_prompt.shape[1]].detach().cpu().numpy())
            allowed_word_ids = allowed_next_toks

        # manually restrict vocab
        #  (allowed_word_ids is list of integers)
        if allowed_word_ids is not None:
            banned_mask = torch.ones(next_token_scores.shape[1])
            banned_mask[allowed_word_ids] = 0
            if data_to_restrict_w is None:
                banned_mask[eos_token_id] = 0
                banned_mask[eos_token_id2] = 0
            banned_mask = banned_mask.bool().unsqueeze(0).to(model.device) # [1 x 50,000]
            next_token_scores = next_token_scores.masked_fill(banned_mask, -float("inf"))

        # Adjust logits, truncate to top p or top k
        #   for the [num seq, vocab_size] it sets the non-top-k tokens to -inf
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += ((outputs.attentions,))
            if output_hidden_states:
                decoder_hidden_states += ((outputs.hidden_states,))

        # Compute next token probs
        probs = nn.functional.softmax(next_token_scores, dim=-1)

        # Printing Examples
        if probs.shape[0]==1:
            top_k = logits_warper[-1].top_k
            probs_detached = probs.detach().cpu().squeeze()
            sort_idx = torch.argsort(probs_detached, dim=-1, descending=True) # can actually do by batch
            top_tokens = sort_idx[0:top_k]
            top_probs = probs_detached[sort_idx][0:top_k] # TODO:
            if verbose:
                print(tokenizer.decode(input_ids[0]))
                print(f'top tokens: {tokenizer.decode(top_tokens)}')
                print(f'p:\t{top_probs.numpy().round(2)}')


        if sample_w_cvar:

            # get next state value distribution
            Vp = []
            Vp_quantiles = []
            for tok in top_tokens:
                tok = tok.unsqueeze(-1).to(model.device)
                if use_prompt_for_dist:
                    input_ids_inner = torch.cat([input_ids, tok[:, None]], dim=-1)
                else:
                    input_ids_inner = torch.cat([input_ids_wo_prompt, tok[:, None]], dim=-1)

                with torch.no_grad():
                    outputs_inner = model(input_ids=input_ids_inner,
                                    attention_mask=torch.ones_like(input_ids_inner),
                                    return_dict=True,
                                    output_attentions=output_attentions,
                                    output_hidden_states=True,
                    )

                states = outputs_inner['hidden_states'][-1][:,-1,:].unsqueeze(1)
                with torch.no_grad():
                    thetas = Z_network(states).detach().cpu().numpy().squeeze()
                if flip_rewards:
                    thetas = (-1.*thetas)[::-1] # [lowest negative score to 0] corresponds to alpha [0, 0.05, .., 1]
                cvars = calc_cvar_from_quantiles(thetas, taus, alphas)
                Vp.append(cvars)
                Vp_quantiles.append(thetas)
            Vp = np.array(Vp)
            Vp_quantiles = np.array(Vp_quantiles) # just for visualizing

            p_distorted, xis, extra = distort_probabilities(top_probs.cpu().numpy(), cvar_alpha, alphas, Vp)

            p_storage.append(list(top_probs.numpy()))
            pd_storage.append(list(p_distorted))
            token_storage.append([tokenizer.decode(tok) for tok in top_tokens])
            cvar_storage.append(Vp)
            quantile_storage.append(Vp_quantiles)
            successes.append(extra['success'])

            assert probs.shape[0]==1
            for idx, pd in zip(sort_idx[0:top_k], p_distorted):
                probs[:,int(idx)]=pd
            if verbose:
                print(f'pd:\t{p_distorted.round(2)}')
                print(f'diff:\t{(p_distorted-top_probs.numpy()).round(2)}')
                print(cvar_alpha)
                print(f'cvar0:\t{Vp[:,0].round(2)}')
                print(f'cvar1:\t{Vp[:,-1].round(2)}')
            if step_by_step:
                import ipdb; ipdb.set_trace()

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        selected_tok_scores = []
        for nti, nt in enumerate(next_tokens):
            selected_tok_scores.append(float(next_token_scores[nti,nt].detach().cpu()))
        selected_tok_scores_all.append(selected_tok_scores)

        # adjust alpha
        if sample_w_cvar and pcvar:
            cvar_alpha = float(xis[top_tokens.cpu().numpy()==next_tokens.detach().cpu().numpy()]*cvar_alpha)
            cvar_alpha = np.max(np.min((cvar_alpha,1)),0)
            alpha_storage.append(cvar_alpha)
        elif sample_w_cvar and not pcvar:
            alpha_storage.append(cvar_alpha)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step (and pass hidden states for more efficient decoding?)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            not_done = torch.logical_and(next_tokens != eos_token_id,
                                         next_tokens != eos_token_id2)
            unfinished_sequences = unfinished_sequences.mul((not_done).long()) # recreates the [1,1,1,0,1] for unfinished sequences

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            #import ipdb; ipdb.set_trace()
            break

    other_outputs = {'alphas': alpha_storage,
                    'p_storage': p_storage,
                    'pd_storage': pd_storage,
                    'token_storage': token_storage,
                    'cvar_storage': cvar_storage,
                    'quantile_storage':quantile_storage,
                    'successes': successes,
                    'selected_tok_scores_all': selected_tok_scores_all}

    if return_dict_in_generate:
        return SampleDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        ), other_outputs
    else:
        return input_ids, other_outputs

def generate(
        model,
        tokenizer,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        eos_token_id2: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        bad_words_ids = None,
        allowed_word_ids = None,
        data_to_restrict_w = None,
        cvar_alpha = 1.0,
        Z_network = None,
        tokenized_prompt = None,
        step_by_step = False,
        use_prompt_for_dist = False,
        flip_rewards = False,
        pcvar = True,
        verbose = True,
        **model_kwargs):

        # Dealing with arguments #
        num_beams = 1
        max_length = max_length if max_length is not None else model.config.max_length
        do_sample = do_sample if do_sample is not None else model.config.do_sample
        num_return_sequences = (num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences)
        eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        eos_token_id2 = eos_token_id2 if eos_token_id2 is not None else model.config.eos_token_id
        output_scores = output_scores if output_scores is not None else model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate)
        pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states


        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        is_sample_gen_mode = do_sample is True
        if is_sample_gen_mode == False:
            raise NotImplementedError

        model_kwargs["use_cache"] = use_cache
        cur_len = input_ids.shape[-1] # Not sure I use

        logits_processor = LogitsProcessorList()
        if min_length is not None:
            logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id=eos_token_id))
            logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id2=eos_token_id2))
        if bad_words_ids is not None:
            logits_processor.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))


        if max_length is not None:
            stopping_criteria = StoppingCriteriaList([
                MaxLengthCriteria(max_length=max_length)
            ])
        else:
            stopping_criteria = StoppingCriteriaList()

        logits_warper = get_logits_warper(
            model, top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        # expand input_ids with `num_return_sequences` additional sequences per batch
        # if just one example input is provided, then the new shape will be [num_return_sequence, input_size]
        input_ids, model_kwargs = expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            **model_kwargs,
        )

        return sample(
            model,
            tokenizer,
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            eos_token_id2=eos_token_id2,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            allowed_word_ids=allowed_word_ids,
            data_to_restrict_w = data_to_restrict_w,
            cvar_alpha = cvar_alpha,
            Z_network = Z_network,
            tokenized_prompt = tokenized_prompt,
            step_by_step = step_by_step,
            use_prompt_for_dist = use_prompt_for_dist,
            flip_rewards = flip_rewards,
            pcvar = pcvar,
            verbose = verbose,
            **model_kwargs,
        )
