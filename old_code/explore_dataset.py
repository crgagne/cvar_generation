from datasets import load_from_disk
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/social_i_qa",
        help="pretrained model name or path to local checkpoint")
    args = parser.parse_args()

    dataset = load_from_disk(args.data)
    dataset = dataset.rename_column('context','text')

    dataset_filtered = dataset.filter(lambda example: 'the doctor' in example['text'].lower())
    for example in dataset_filtered['train']['text']:
        print(example)


if __name__ == '__main__':

    main()
