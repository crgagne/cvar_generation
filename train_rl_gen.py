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
from generator import generate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2/",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_seq", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="<|endoftext|>The")
    parser.add_argument("--restrict", type=str, default="False")

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
    model.config.pad_token_id = model.config.eos_token_id # otherwise it generates ....
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    encoded_prompt = tokenizer(args.prompt, return_tensors='pt').to(device)

    sid_obj = SentimentIntensityAnalyzer()

    allowed_words = np.load('mscl/top_single_tok_english_words.npy',allow_pickle=True).tolist()
    #allowed_words =[]
    allowed_words.extend(['The', 'A',
                                'dog', 'cat', 'man','woman',
                                'old','with', #'who',
                                'is', 'are', 'was', 'has',
                                'ran','ate','walked', 'thought','chased',
                                'to','under','the', 'a','about','through',
                                'tree','river','fire','death','apple','orange','gun',
                                'horrible','hates',
                                'other','same',
                                'amazing','great','loves'])

    allowed_word_ids = tokenizer(allowed_words,add_prefix_space=True).input_ids
    allowed_word_ids.extend(tokenizer(['The', 'A'],add_prefix_space=False).input_ids)
    for aw in allowed_word_ids:
       assert len(aw)==1
    allowed_word_ids = [aw[0] for aw in allowed_word_ids]
    if eval(args.restrict)==False:
        allowed_word_ids = None

    # train
    model.train_value_head_only()

    # set up optimizers, and learning rate schedule
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate) # TODO: consider whether I need just a subset of parameters..

    for epoch in range(args.epochs):
        for idx in tqdm(range(args.num_seq), leave=True, position=0):

            # generate data
            generated_batch_toks = generate(model, tokenizer,
                                input_ids=encoded_prompt['input_ids'],
                                attention_mask=encoded_prompt['attention_mask'],
                                max_length=20, num_beams = 1,
                                temperature=1., num_return_sequences=10,
                                do_sample=True, eos_token_id=13,
                                bad_words_ids = None,
                                top_k = 0, top_p=0.95, # this will help it be more grammatical, but it may also not be able to say anything
                                allowed_word_ids = allowed_word_ids,
                                )

            decoded = tokenizer.batch_decode(generated_batch_toks)
            reencoded_batch = tokenizer(decoded, return_tensors='pt', padding=True).to(device)

            # maybe remove the prompt .. 


            # score generations
            rewards = [sid_obj.polarity_scores(sentence)['compound'] for sentence in decoded]
            #import ipdb; ipdb.set_trace()

            # train the value function
            output = model(input_ids=reencoded_batch['input_ids'],
                           labels=reencoded_batch['input_ids'],
                           attention_mask=reencoded_batch['attention_mask'],
                           rewards=torch.Tensor(rewards).to(device))

            _, td_loss = output[0], output[2]
            values = output[3].detach().cpu().numpy()

            td_loss.backward() # consider the grad clipping in the main.py for the DQN.
            optimizer.step()

            # switch value target heads every batch
            if idx % int(args.num_seq*0.1)==0:
                model.value_head_target.load_state_dict(model.value_head.state_dict())

            # log
            if idx % int(args.num_seq*0.25)==0:


                print(decoded)
                test_example = tokenizer('love amazing success.', return_tensors='pt').to(device)
                print(test_example)
                output = model(input_ids=test_example['input_ids'].to(device),
                               attention_mask=test_example['attention_mask'].to(device))
                print(output[2])

                test_example = tokenizer('hate terrible failure.', return_tensors='pt').to(device)
                print(test_example)
                with torch.no_grad():
                    output = model(input_ids=test_example['input_ids'],attention_mask=test_example['attention_mask'])
                print(output[2])


if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=3 python train_rl_gen.py --model models/pretrained/gpt2 --prompt "<|endoftext|>The" --restrict True
    # CUDA_VISIBLE_DEVICES=3 python train_rl_gen.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000 --prompt "<|endoftext|>The" --restrict True
    # CUDA_VISIBLE_DEVICES=3 python train_rl_gen.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000 --prompt "<|endoftext|>The" --restrict False
    # CUDA_VISIBLE_DEVICES=3 python train_rl_gen.py --model models/pretrained/gpt2-large --prompt "<|endoftext|>The man went to the park. The man cleaned up his house. The man ate dinner."
