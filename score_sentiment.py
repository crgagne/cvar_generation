

from generate_sentences import score_sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import datasets
from tqdm import tqdm
import argparse
import torch
import numpy as np


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="go_emotions")
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    sentiment_tokenizer = AutoTokenizer.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('models/pretrained/cardiffnlp-twitter-roberta-base-sentiment')
    sentiment_model.to(device);

    if args.data=='go_emotions':
        data_go = datasets.load_from_disk('data/raw/go_emotions/')

        sentences = []
        for split in ['train','test','validation']:
            sentences.extend(data_go[split]['text'])

        save_folder = 'data/results/single_sentences_IYou_3_emo/'
        file_full = 'go_full_generations.txt'
        file_ends = 'go_ends.txt'

        name_full=save_folder + file_full
        name_ends=save_folder + file_ends
        with open(name_full, 'w') as file_full, open(name_ends, 'w') as file_end:
            i = 0
            for sentence in tqdm(sentences):
                sentence = sentence.strip()
                if sentence[-1] not in ['.','?','!']:
                    sentence+='.'

                try:
                    # score sentiment
                    sentiment = score_sentiment(sentence, sentiment_tokenizer, sentiment_model, device)

                    sentence+=' r='+str(np.round(sentiment[0],2))+'\n'
                    if len(sentence.split(' '))>3:
                        file_full.write(sentence)
                        file_end.write(sentence)

                    i+=1
                    # if i>100:
                    #     break
                except:
                    print('failed: '+sentence)


if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=2 python score_sentiment.py
