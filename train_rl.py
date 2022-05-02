from pathlib import Path
import argparse
from helpers import set_seeds
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
)
from models import GPT2CustomDoubleHeadsModel
import os
from scipy.special import softmax
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm

# from torch.utils.data.dataloader import default_collate this doesn't work with hugging face dataset
from transformers import default_data_collator
from batch_datasets import get_batch_dataset

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/finetuned/gpt2/social_i_qa/checkpoint-1000",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    #args.data = args.data.replace('{model}',args.model.replace('/','_'))

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    set_seeds(seed = args.seed)

    # load model
    config = GPT2Config.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2CustomDoubleHeadsModel.from_pretrained(args.model, config=config)
    model.to(device)

    # load dataset
    dataset = get_batch_dataset(args.data)

    # tokenize
    def tokenize(batch):
       return tokenizer(batch['text'], truncation=True, max_length=50, padding='max_length')
    dataset = dataset.map(tokenize, num_proc=1, batched=True)

    # tensorize sentiment
    def tensorize(batch):
        batch['sentiment']=[torch.Tensor([sentiment]) for sentiment in batch['sentiment']]
        return(batch)
    dataset = dataset.map(tensorize, num_proc=1, batched=True)

    def collate_with_strings(batch, str_column = 'text'):
        new_batch = []; strings = []
        for _batch in batch:
            strings.append(_batch[str_column])
            _batch.pop(str_column, None)
            new_batch.append(_batch)
        return default_data_collator(new_batch), strings

    train_data = DataLoader(dataset, collate_fn=collate_with_strings, batch_size=args.batch_size, shuffle=True)

    # train
    model.train_value_head_only()

    # set up optimizers, and learning rate schedule
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate) # TODO: consider whether I need just a subset of parameters..

    for epoch in range(args.epochs):
        for idx, (batch, text) in tqdm(enumerate(train_data, start=1), leave=True, position=0):
            optimizer.zero_grad()
            output = model(input_ids=batch['input_ids'].to(device),
                           labels=batch['input_ids'].to(device),
                           attention_mask=batch['attention_mask'].to(device),
                           rewards=batch['sentiment'].to(device))
            _, td_loss = output[0], output[1]
            values = output[3].detach().cpu().numpy()

            td_loss.backward() # consider the grad clipping in the main.py for the DQN.
            optimizer.step()

            if idx % int(len(train_data)*0.25)==0:

                # log
                print(f'\ntdloss:{td_loss:.3f}\n')
                print(np.round(values[0,:],2))
                print(batch['sentiment'][0])
                print('')

                # switch value target heads every epoch
                model.value_head_target.load_state_dict(model.value_head.state_dict())

    import ipdb; ipdb.set_trace()
    test_example = tokenizer('A badger', return_tensors='pt').to(device)
    print(test_example)
    output = model(input_ids=test_example['input_ids'].to(device),
                   attention_mask=test_example['attention_mask'].to(device))
    print(output[2])

    test_example = tokenizer('The badger', return_tensors='pt').to(device)
    print(test_example)
    with torch.no_grad():
        output = model(input_ids=test_example['input_ids'],attention_mask=test_example['attention_mask'])
    print(output[2])



if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=3 python train.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000
