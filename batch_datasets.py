from datasets import load_dataset, concatenate_datasets
import argparse
from pathlib import Path
import numpy as np

def get_batch_dataset(dataset_name):

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", type=str, default="The/A",
    #     help="pretrained model name or path to local checkpoint")
    # args = parser.parse_args()

    if dataset_name=='The/A':
        # load dataset
        dataset_loader_path = Path(__file__).parent / 'dataset_builder.py'
        dataset_positive = load_dataset(str(dataset_loader_path),
                               data_filepath="data/results/generations/models_finetuned_gpt2_social_i_qa_checkpoint-1000/exploring_pretrained_gpt2_<|endoftext|>The.txt")
        dataset_negative = load_dataset(str(dataset_loader_path),
                               data_filepath="data/results/generations/models_finetuned_gpt2_social_i_qa_checkpoint-1000/exploring_pretrained_gpt2_<|endoftext|>A .txt")
        dataset_positive = dataset_positive['train']
        dataset_negative = dataset_negative['train']
        def make_positive(example):
            example['sentiment']=1.
            return(example)
        def make_negative(example):
            example['sentiment']=-1. # using positive
            return(example)
        dataset_positive = dataset_positive.map(make_positive)
        dataset_negative = dataset_negative.map(make_negative)
        #dataset_positive = dataset_positive.filter(lambda example: example['sentiment']>0)
        #dataset_negative = dataset_negative.filter(lambda example: example['sentiment']<=0)
        dataset = concatenate_datasets([dataset_positive, dataset_negative])

    else:

        dataset_loader_path = Path(__file__).parent / 'dataset_builder.py'
        dataset = load_dataset(str(dataset_loader_path),
                               data_filepath=dataset_name)

        dataset=dataset['train']
        
    return(dataset)


if __name__ == '__main__':

    main()

    # python create_batch_dataset.py
