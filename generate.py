from pathlib import Path
import argparse
from helpers import set_seeds
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from models import GPT2CustomDoubleHeadsModel
import os
from scipy.special import softmax
import torch
import numpy as np

from generator import generate

# here I'll have different ways of running the model
# - sample (normal)
# - sample (using CVaR)

# /home/cgagne/cvar_generation/conda_env/lib/python3.9/site-packages/transformers/


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/gpt2",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument('--gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument("--seed", type=int, default=2311)
    parser.add_argument("--num_seq", type=int, default=100)
    parser.add_argument("--save_name", type=str, default='exploring_pretrained_gpt2')
    parser.add_argument("--score", type=str, default='False')
    parser.add_argument("--scorer", type=str, default="models/pretrained/nlptown-bert-base-multilingual-uncased-sentiment") #"models/pretrained/cardiffnlp-twitter-roberta-base-sentiment"
    parser.add_argument("--prompt", type=str, default="<|endoftext|>The")
    parser.add_argument("--restrict", type=str, default="False")
    args = parser.parse_args()

    #set_seeds(seed = args.seed)

    if args.gpus>0:
        device=f"cuda:{torch.cuda.current_device()}"
    else:
        device='cpu'

    # load model
    config = GPT2Config.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    #model = GPT2CustomDoubleHeadsModel.from_pretrained(args.model, config=config)
    model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    # set_seeds(seed = args.seed)
    # output = model.generate(input_ids=inputs['input_ids'],
    #                         attention_mask=inputs['attention_mask'],
    #                         max_length=20, num_beams = 1,
    #                         temperature=1., num_return_sequences=args.num_seq,
    #                         do_sample=True, eos_token_id=13) #length_penalty=0.5) # top_p=0.95,
    #                                 #) #diversity_penalty; repetition_penalty=1.0, forced_eos_token_id=13


    allowed_words = np.load('mscl/top_single_tok_english_words.npy',allow_pickle=True).tolist()
    #allowed_words = []
    allowed_words.extend(['The', 'A',
                                'dog', 'cat', 'man','woman',
                                'old','with', #'who',
                                'is', 'are', 'was', 'has',
                                #'and','but',
                                'ran','ate','walked', 'thought','chased',
                                'to','under','the', 'a','about','through',
                                'tree','river','fire','death','apple','orange','gun',
                                'horrible','great','other','same'])

    allowed_word_ids = tokenizer(allowed_words,add_prefix_space=True).input_ids
    allowed_word_ids.extend(tokenizer(['The', 'A'],add_prefix_space=False).input_ids)
    for aw in allowed_word_ids:
       assert len(aw)==1
    allowed_word_ids = [aw[0] for aw in allowed_word_ids]
    if eval(args.restrict)==False:
        allowed_word_ids = None

    import ipdb; ipdb.set_trace()

    output = generate(model, tokenizer,
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=50, num_beams = 1,
                        temperature=1, num_return_sequences=args.num_seq,
                        do_sample=True, eos_token_id=13,
                        bad_words_ids = None,
                        top_k = 0, top_p=0.95, # this will help it be more grammatical, but it may also not be able to say anything
                        allowed_word_ids = allowed_word_ids,
                        )

    # make save path
    save_path = str(Path(__file__).parent) + '/' + "data" + '/' + "results" + '/' + "generations" + '/' + args.model.replace('/','_')
    print(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # decode outputs
    decoded_list = []
    for i in range(args.num_seq):
        decoded = tokenizer.decode(output[i], skip_special_tokens=False)
        decoded = decoded.replace('\n','')
        decoded = decoded.replace('..','')
        decoded = decoded.replace('<|endoftext|>','')
        decoded_list.append(decoded)


    if eval(args.score):
        scorer_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
        scorer_model = AutoModelForSequenceClassification.from_pretrained(args.scorer)
        scorer_model.to(device)

        # analyze sentiment
        scores=[]
        for decoded in decoded_list:
            output = scorer_model(**scorer_tokenizer(decoded, return_tensors='pt').to(device))
            probs = softmax(output[0][0].detach().cpu().numpy()) # these are negative,neutral, positive.
            if len(probs)==5:
                score = np.dot(probs,np.arange(-2,3))
            else:
                score = np.dot(probs,np.arange(-1,1))
            scores.append(score)

    # sorting
    sort_idx = np.argsort(scores)
    scores = [scores[i] for i in sort_idx]
    decoded_list = [decoded_list[i] for i in sort_idx]

    # save results
    print(save_path)
    with open(save_path / Path(args.save_name+'_'+args.prompt+'.txt'), 'w') as f:
        for decoded, score in zip(decoded_list, scores):
            line = f"{decoded} r={score:.3f} \n"
            print(line)
            #f.write(line)


if __name__ == '__main__':

    main()

    # python generate.py --model models/pretrained/gpt2/
    # python generate.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000
    # CUDA_VISIBLE_DEVICES=3 python generate.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000 --score True --prompt "<|endoftext|>The dog" --num_seq 10
    # CUDA_VISIBLE_DEVICES=3 python generate.py --model models/pretrained/gpt2 --score True --prompt "<|endoftext|>The" --num_seq 2

    # CUDA_VISIBLE_DEVICES=3 python generate.py --model models/pretrained/gpt2 --score True --prompt "The man" --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/finetuned/gpt2/social_i_qa/checkpoint-1000 --score True --prompt "The man" --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/finetuned/gpt2/social_i_qa/checkpoint-1000 --score True --prompt "The" --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2 --score True --prompt "<|endoftext|>The" --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2 --score True --prompt "<|endoftext|>" --num_seq 1 --restrict True

    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2 --score True --prompt "<|endoftext|>The man went to the park. The man cleaned up his house. The man ate dinner." --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2 --score True --prompt "<|endoftext|>The man went to the park. The man cleaned up his house. The man ate dinner." --num_seq 10 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The man went to the park. The man cleaned up his house. The man ate dinner." --num_seq 10 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The man went to the park. The man cleaned up his house. The man ate dinner." --num_seq 10 --restrict False

    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The man went to the park. The woman cleaned up her house." --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The" --num_seq 1 --restrict True
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The" --num_seq 1 --restrict True

    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>The man went to the park. The woman cleaned up her house." --num_seq 1 --restrict False
    # CUDA_VISIBLE_DEVICES=3 python generate.py  --model models/pretrained/gpt2-large --score True --prompt "<|endoftext|>I have a math test. I went to the doctors. I made dinner." --num_seq 10 --restrict False
