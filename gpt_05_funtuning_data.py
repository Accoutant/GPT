from datasets import load_dataset
from tokenizers import Tokenizer
import re
from gpt_01_process_data import my_collate
from torch.utils.data import DataLoader
import pickle

tokenizer = Tokenizer.from_file('token.json')
tokenizer.enable_padding(length=30)
tokenizer.enable_truncation(max_length=30)

dataset = load_dataset('text', data_files={"train": "data/train_150k.txt", "test": "data/test_62k.txt"},
                       cache_dir='./data')


def split_input_and_target(example):
    """区分数据集的input和target"""
    target, input = example['text'].split("\t")
    if "@" in input:
        input = re.sub("(@.*? )", "", input)
        input = input.rstrip(" ")
    valid_lens = len(input.split())
    return {"input": input, "target": target, "valid_lens": valid_lens}


def tokenize(example):
    return {"input_ids": tokenizer.encode(example['input']).ids,
            "target_ids": int(example['target'])}


if __name__ == "__main__":
    train_dataset = dataset['train'].map(split_input_and_target, num_proc=4, remove_columns='text')
    test_dataset = dataset['test'].map(split_input_and_target, num_proc=4, remove_columns='text')
    train_dataset = train_dataset.map(tokenize, num_proc=4, remove_columns=['input', 'target'])
    test_dataset = test_dataset.map(tokenize, num_proc=4, remove_columns=['input', 'target'])

    train_dataset = train_dataset.filter(lambda x: x['valid_lens'] != 0)
    test_dataset = test_dataset.filter(lambda x: x['valid_lens'] != 0)
    # 去除有效长度为0的数据，防止出现nan
    tran_iter = DataLoader(train_dataset, shuffle=True, collate_fn=my_collate, batch_size=32)
    test_iter = DataLoader(test_dataset, shuffle=True, collate_fn=my_collate, batch_size=32)

    with open("fine_tunning.pkl", 'wb') as f:
        pickle.dump((tokenizer, tran_iter, test_iter), f)




