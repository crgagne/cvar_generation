import warnings
from os import path

from transformers import (
    AutoTokenizer,
    AutoModel,
    BertForMaskedLM,
    BertTokenizer,
    AutoModelForSequenceClassification,
)

def download_and_save(modelname):

    warnings.warn("Using Auto* Methods to guess model")
    if 'sentiment' in modelname:
        model = AutoModelForSequenceClassification.from_pretrained(modelname) # otherwise it doesn't guess the right model; that's dumb
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


def main():

    #download_and_save("gpt2")
    #download_and_save("cardiffnlp/twitter-roberta-base-sentiment")
    #download_and_save("nlptown/bert-base-multilingual-uncased-sentiment")
    download_and_save("gpt2-large")


if __name__ == '__main__':
    main()
