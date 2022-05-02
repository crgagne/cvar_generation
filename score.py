from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained/cardiffnlp-twitter-roberta-base-sentiment")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)




if __name__ == '__main__':

    main()

    # python generate.py --model models/pretrained/gpt2/
    # python generate.py --model models/finetuned/gpt2/social_i_qa/checkpoint-1000
