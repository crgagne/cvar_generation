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
from transformers import AutoTokenizer, AutoConfig


def predict(sentence, emo_model, emo_tokenizer, device, threshold=0.3, verbose=False):
    inputs = emo_tokenizer(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = emo_model(**inputs)
        probs = outputs[0].float().sigmoid().cpu().numpy().squeeze()
        probs_threshed = (probs>threshold).squeeze()
        emotions = [emo_model.config.id2label[idd] for idd in np.where(probs_threshed)[0]]
    if verbose:
        print(emotions)
    return(probs, probs_threshed, emotions)

def create_emotion_reward_strings(line_full, line_end, emo_model, probs, score_df = None):

    line_full = line_full.replace('\n', '')
    line_end = line_end.replace('\n', '')

    for i in range(28):
        label = emo_model.config.id2label[i]
        appendie = f',r_{label}={probs[i]:0.2f}'
        line_full+=appendie
        line_end+=appendie
        if score_df is not None:
            score_df.loc[l,label]=probs[i]

    line_full+='\n'
    line_end+='\n'

    return(line_full, line_end, score_df)

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/monologg-bert-base-cased-goemotions-original/")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--save_folder", type=str, default='single_sentences_IYou_3_emo')
    parser.add_argument("--file_ends", type=str, default='ends.txt')
    parser.add_argument("--file_full", type=str, default='full_generations.txt')

    args = parser.parse_args()
    args.save_folder = Path(__file__).parent / 'data' / 'results' / args.save_folder
    print(f'save folder: {args.save_folder}')

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model
    emo_model = BertForMultiLabelClassification.from_pretrained(args.model)
    emo_tokenizer = AutoTokenizer.from_pretrained(args.model)
    emo_labels = [emo_model.config.id2label[i] for i in range(28)]
    emo_model.to(device)

    # load data for reading
    file_ends = open(args.save_folder / args.file_ends, 'r')
    file_full = open(args.save_folder / args.file_full, 'r')
    lines_end = file_ends.readlines()
    lines_full = file_full.readlines()

    # score emotions
    score_df = pd.DataFrame(columns = ['full', 'end', 'sentiment']+[emo_model.config.id2label[i] for i in range(28)])
    new_full = []
    new_ends = []
    l = 0
    iterr = zip(lines_end, lines_full)
    for line_end, line_full in tqdm(iterr, total=len(lines_end)):

        sentiment = float(line_end.replace('\n','').split('r=')[-1])

        score_df.loc[l,'full']=line_full
        score_df.loc[l,'end']=line_end
        score_df.loc[l,'sentiment']=sentiment

        # score
        probs, probs_threshed, emotions = predict(line_end, emo_model, emo_tokenizer, device)

        line_full, line_end, score_df = create_emotion_reward_strings(line_full, line_end, emo_model, probs, score_df = score_df)

        new_full.append(line_full)
        new_ends.append(line_end)

        l+=1
        # if l>1000:
        #     break

    # save results
    name_full=args.save_folder / args.file_full.replace('.txt','_w_emotions.txt')
    name_ends=args.save_folder / args.file_ends.replace('.txt','_w_emotions.txt')
    with open(name_full, 'w') as file_full, open(name_ends, 'w') as file_end:
        for line_full, line_end in zip(new_full, new_ends):
            file_full.write(line_full)
            file_end.write(line_end)

    if 'go' in args.file_full:
        score_df.to_csv(args.save_folder / 'go_emotion_df.csv')
    else:
        score_df.to_csv(args.save_folder / 'emotion_df.csv')

if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=0 python score_emotions.py --file_ends 'ends.txt' --file_full 'full_generations.txt'
    # CUDA_VISIBLE_DEVICES=0 python score_emotions.py --file_ends 'go_ends.txt' --file_full 'go_full_generations.txt'
