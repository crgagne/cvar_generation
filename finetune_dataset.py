import argparse
from pathlib import Path
from helpers import set_seeds
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
)
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import os
from transformers import TrainingArguments

try:
    MAX_NUM_PROC = int(os.environ["SLURM_CPUS_ON_NODE"])
except KeyError:
    MAX_NUM_PROC = len(os.sched_getaffinity(0))

from trainer import CustomTrainer

# basic code for finetuning gpt2 etc.
# https://github.com/huggingface/transformers/blob/27c1b656cca75efa0cc414d3bf4e6aacf24829de/examples/run_lm_finetuning.py

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_v1')
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2-large")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--subsample", action='store_true') # way too long if I don't do this
    args = parser.parse_args()

    save_dir = Path("models/finetuned/") / Path(args.model).name / Path(args.dataset).name
    print(save_dir)

    if args.gpus>0:
        device='cuda:0'
    else:
        device='cpu'

    set_seeds(seed = args.seed)

    # load model add new tokens
    config = GPT2Config.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|generate|>","<|score|>"
    ]})
    model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # load dataset
    dataset = load_from_disk(args.dataset)
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].rename_column('post', 'text')
        columns_to_remove = [c for c in dataset[split].column_names if c not in ['text']]
        dataset[split] = dataset[split].remove_columns(columns_to_remove)
        if args.subsample:
            if split=='train':
                dataset[split] = dataset[split].select(range(50_000))
            else:
                dataset[split] = dataset[split].select(range(5_000))

    # tokenize
    def tokenize(batch):
        tokenized = tokenizer(batch["text"], max_length=50, padding='max_length',truncation=True)
        return tokenized
    dataset = dataset.map(tokenize, num_proc=1, batched=True)
    dataset = dataset.remove_columns(['text'])

    # set up dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors='pt') #padding='longest', eturn_tensors='pt',

    # print example data
    out = data_collator([dataset['train'][i] for i in range(5)])
    for key in out:
       print(f"{key} shape: {out[key].shape}")

    # https://huggingface.co/course/chapter7/6?fw=pt
    args = TrainingArguments(
        output_dir=save_dir,
        logging_dir=save_dir,
        logging_strategy='steps',
        logging_first_step=True,
        per_device_train_batch_size=2, #
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=20,
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
        save_steps=200,
        fp16=True,
        push_to_hub=False,
        )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print('... training')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    assert trainer.args._n_gpu <= 1
    trainer.train()


if __name__ == '__main__':

    main()

    # CUDA_VISIBLE_DEVICES=1 python finetune_dataset.py --gpus 1 --subsample
