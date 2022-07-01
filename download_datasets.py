from datasets import load_dataset
from os import path


def download_and_save(dataset_name, specs = None):

    save_dir = path.join(path.dirname(__file__), "data", "raw", dataset_name)
    dataset = load_dataset(dataset_name, specs)
    dataset.save_to_disk(save_dir)


def main():

    #download_and_save('social_i_qa') # https://huggingface.co/datasets/go_emotions (there's also a 'raw' version which is not split)
    download_and_save('go_emotions')


if __name__ == '__main__':
    main()
