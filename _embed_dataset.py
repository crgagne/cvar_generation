from pathlib import Path
import argparse
from helpers import set_seeds

import os
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from emotion_models import BertForMultiLabelClassification
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_from_disk


@torch.no_grad()
@torch.no_grad()

def embed(text, model, tokenizer, method="mean"):
    assert method in ("mean", "cls"), f"{method} not implemented; method must be 'mean' or 'cls'"

    inputs = tokenizer(text, return_tensors="pt", padding="longest",max_length=85, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs["hidden_states"][-1]
    #import pdb; pdb.set_trace()


    attention_mask = features["attention_mask"].unsqueeze(2)
    hidden_states = hidden_states * attention_mask  # don't average over pad tokens
    # hidden_states = hidden_states[:, 1:, :]  # exclude cls token? sep?
    mean_hidden_states = torch.mean(hidden_states, dim=1)
    mean_hidden_states = list(mean_hidden_states.unbind())
    return {"embedding": mean_hidden_states}


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_v1_w_emosent')
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2-large")
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model for calculating hidden states (add extra token for score)
    config = GPT2Config.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
    model.to(device)

    # load dataframe
    dataset = load_from_disk(args.dataset)

    dataset = dataset['train'].select(range(100_000))

    def embed_batch(rows):
        probs = embed(rows["post"], emo_model, emo_tokenizer, device)
        sentiments = score_sentiment(rows["post"], sentiment_model, sentiment_tokenizer, device)
        results = {}
        results['sentiment']=sentiments
        for emotion in emo_model.config.label2id.keys():
            id = emo_model.config.label2id[emotion]
            results[emotion]=probs[:,id]
        return results

    dataset = dataset.map(score_emotions_batch, batched=True,
                          batch_size=20, num_proc=1)

    dataset.save_to_disk(args.dataset+'_w_emosent')


if __name__ == '__main__':

    main()
