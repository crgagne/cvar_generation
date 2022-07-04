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

EMOTIONS = ['admiration','amusement','anger','annoyance','approval','caring',
 'confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment',
 'excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride',
 'realization','relief','remorse','sadness','surprise','neutral']

def remove_codes(sentences):
    return [s.replace("<|generate|>",'').replace("<|score|>","") for s in sentences]

@torch.no_grad()
def score_emotions(sentences, emo_model, emo_tokenizer, device, threshold=0.3, verbose=False):
    sentences = remove_codes(sentences)
    inputs = emo_tokenizer(sentences, return_tensors='pt', padding='longest').to(device)
    outputs = emo_model(**inputs)
    probs = outputs[0].float().sigmoid().cpu().numpy().squeeze()
    return probs

@torch.no_grad()
def score_sentiment(sentences, sentiment_model, sentiment_tokenizer, device):
    sentences = remove_codes(sentences)
    inputs = sentiment_tokenizer(sentences, return_tensors='pt', padding='longest').to(device)
    output = sentiment_model(**inputs)
    probs = softmax(output[0].detach().cpu().numpy(),axis=1)
    if probs.shape[1]==5:
        scores = np.dot(probs,np.arange(-2,3))
    else:
        scores = np.dot(probs,np.arange(-1,2))
    return scores

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_2M')
    parser.add_argument("--emotion_model", type=str, default='models/pretrained/monologg-bert-base-cased-goemotions-original')
    parser.add_argument("--sentiment_model", type=str, default='models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--subsample", action='store_true')
    args = parser.parse_args()

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model
    emo_model = BertForMultiLabelClassification.from_pretrained(args.emotion_model)
    emo_tokenizer = AutoTokenizer.from_pretrained(args.emotion_model)
    emo_labels = [emo_model.config.id2label[i] for i in range(28)]
    emo_model.to(device)

    sentiment_tokenizer = AutoTokenizer.from_pretrained(args.sentiment_model)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    # load dataframe
    dataset = load_from_disk(args.dataset)

    if args.subsample:
        for split in ['train', 'validation', 'test']:
            dataset[split] = dataset[split].select(range(10_000))

    if 'sentences' in args.dataset:
        textname = 'sentence'
    else:
        textname = 'post'

    def score_emotions_batch(rows):
        try:
            probs = score_emotions(rows[textname], emo_model, emo_tokenizer, device)
            sentiments = score_sentiment(rows[textname], sentiment_model, sentiment_tokenizer, device)
        except:
            probs = np.zeros((len(rows[textname]),len(EMOTIONS)))
            sentiments = np.zeros(len(rows[textname]))#*np.nan
        results = {}
        results['sentiment']=sentiments
        for emotion in emo_model.config.label2id.keys():
            id = emo_model.config.label2id[emotion]
            results[emotion]=probs[:,id]
        return results

    dataset = dataset.map(score_emotions_batch, batched=True,
                          batch_size=20, num_proc=1)

    if args.subsample:
        dataset.save_to_disk(args.dataset+'_w_emosent_subsample')
    else:
        dataset.save_to_disk(args.dataset+'_w_emosent')


if __name__ == '__main__':

    main()

    # small datasets
    # CUDA_VISIBLE_DEVICES=1 python score_dataset_emotions_and_sentiment.py --dataset '/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_2M' --subsample
    # CUDA_VISIBLE_DEVICES=2 python score_dataset_emotions_and_sentiment.py --dataset '/home/cgagne/cvar_generation/data/preprocessed/SMHD_sentences_depctrl_2M' --subsample
