from datasets import load_dataset
from tokenizers import Tokenizer
import re

tokenizer = Tokenizer.from_file('token.json')

dataset = load_dataset('text', data_files={"train": "data/train_150k.txt", "test": "test_62k.txt"})


def split_input_and_target(example):
    target, input = example['text'].split("\t")
    if "@" in target:
        input = re.sub("(@.*?) ", "", input)
        input = input.rstrip(" ")
    return {"input": input, "target": target}


if __name__ == "__main__":
    train_dataset = dataset['train'].map(split_input_and_target, num_proc=4, remove_columns='text')
    print(train_dataset[3])