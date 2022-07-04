from datasets import load_from_disk
import argparse
import sys


def replace_tags_and_clean(row):
    post = row['post']
    post = post.replace("<|depression|>","").replace("<|endoftext|>","").replace("<|control|>","")
    post = post.replace("\n"," ").replace("\r"," ")
    post = post.replace("&gt;"," ").replace("\r"," ")
    post = post.strip()
    #post = "<|generate|>"+post+"<|score|>"
    #post = post+'<|endoftext|>'
    return({"post": post})

def split_sentences(rows):
    sentences = []
    for post in rows['post']:
        sents = post.split('. ')
        sents = process_sentences(sents)
        sentences.extend(sents)
    rows['post']=sentences
    return(rows)

def process_sentences(sents):
    sents = [s.strip() for s in sents if len(s.split(' '))>5] # look for sentences of at least 5 words
    sents = [s for s in sents if len(s)>0 ] # remove empty strings
    sents = [s for s in sents if s[-1] not in ['?','!'] ] # remove empty strings
    sents = [s + '.' if s[-1]!='.' else s for s in sents] # add periods if they don't have them
    sents = [s for s in sents if s.count('.')==1] # remove sentences with multiple periods
    return(sents)

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting_dataset", type=str, default='/home/cgagne/cvar_generation/data/raw/SMHD_all_depression+control_64M/')
    parser.add_argument("--ending_dataset", type=str, default='/home/cgagne/cvar_generation/data/preprocessed/SMHD_posts_depctrl_64M')
    parser.add_argument("--subset", action='store_true')
    args = parser.parse_args()

    sys.path.append(args.starting_dataset)

    # load dataset
    dataset = load_from_disk(args.starting_dataset)
    print(dataset)

    # subsample for inspecting the dataset
    if args.subset:
        dataset = dataset.select(range(10_000))
        extra_savename='_test'
    else:
        extra_savename=''

    if args.starting_dataset=='/home/cgagne/cvar_generation/data/raw/SMHD_all_depression+control/':
        for split, sub in zip(['train','validation','test'],[1e7,1e5,1e5]):
            dataset[split] = dataset[split].select(range(int(sub)))
        extra_savename=''
        args.ending_dataset = args.ending_dataset.replace('_64M', '_10M')

    #import ipdb; ipdb.set_trace()

    # filte to length
    print('')
    print('filtering ...')
    dataset = dataset.filter(lambda x: (x["length"]<128) & (x["length"]>5), num_proc=64)
    print(dataset)

    # remove tags
    dataset = dataset.map(replace_tags_and_clean, num_proc=64, batched=False)

    # sort by length for effective batching
    #dataset = dataset.sort("length")

    # remove unneccessary columns
    for split in ['train','validation','test']:
        columns_to_remove = [c for c in dataset[split].column_names if c not in ['post']]
        dataset[split] = dataset[split].remove_columns(columns_to_remove)

    # save
    dataset.save_to_disk(args.ending_dataset+extra_savename)

    # make sentence dataset
    dataset = dataset.map(split_sentences, num_proc=64, batched=True)
    for split in ['train','validation','test']:
        dataset[split] = dataset[split].rename_column('post', 'sentence')

    # save another
    dataset.save_to_disk(args.ending_dataset.replace('posts','sentences')+extra_savename)



if __name__ == '__main__':

    main()
