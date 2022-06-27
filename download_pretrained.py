import warnings
from os import path

from transformers import (
    AutoTokenizer,
    AutoModel,
    BertForMaskedLM,
    BertTokenizer,
    AutoModelForSequenceClassification,
    GPTJForCausalLM,
    GPT2Tokenizer,
    OPTForCausalLM
)

from transformers import GPTJForCausalLM
import torch

def download_and_save(modelname, half=False):

    warnings.warn("Using Auto* Methods to guess model")

    if 'sentiment' in modelname:
        model = AutoModelForSequenceClassification.from_pretrained(modelname) # otherwise it doesn't guess the right model; that's dumb
        tokenizer = AutoTokenizer.from_pretrained(modelname)
    elif 'gpt-j' in modelname:
        model = GPTJForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(modelname)
    elif 'opt' in modelname:
        if half:
            import ipdb; ipdb.set_trace() # I don't know if this works
            model = OPTForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
        else:
            model = OPTForCausalLM.from_pretrained(modelname)

        tokenizer = GPT2Tokenizer.from_pretrained(modelname)
    else:
        model = AutoModel.from_pretrained(modelname)
        tokenizer = AutoTokenizer.from_pretrained(modelname)

    # remove extra path information
    if '/' in modelname:
        modelname = modelname.replace('/','-')

    save_path = path.join(path.dirname(__file__), "models", "pretrained", modelname)
    print(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # remove from cache afterwards...


def main():

    #download_and_save("gpt2")
    #download_and_save("cardiffnlp/twitter-roberta-base-sentiment")
    #download_and_save("nlptown/bert-base-multilingual-uncased-sentiment")
    #download_and_save("gpt2-large")
    #download_and_save("EleutherAI/gpt-j-6B")
    #download_and_save("facebook/opt-350m")
    download_and_save("facebook/opt-2.7b", half=True)
    #download_and_save("facebook/opt-1.3b")

if __name__ == '__main__':
    main()
