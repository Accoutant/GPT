from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
import pickle
import torch


# 加载数据集
datasets = load_dataset("bookcorpus.py", split='train[:1000000]', num_proc=4, cache_dir='./data_cache', )
test_datasets = load_dataset("bookcorpus.py", split='train[1000000:1010000]', num_proc=4, cache_dir='./data_cache')

# 加载tokenizer
tokenizer = Tokenizer(BPE(unk_token='<unk>'))
trainer = BpeTrainer(special_tokens=['<pad>', '<unk>', '<cls>', '<sep>'])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(datasets['text'], trainer)
tokenizer.post_processor = TemplateProcessing(single='<cls> $A',
                                              special_tokens=[('<cls>', 2), ('<sep>', 3)])
tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=31)     # 因为设置input_ids和target_ids要cut
tokenizer.enable_truncation(max_length=31)

tokenizer.save('token.json')


def get_input_and_target(example):
    example['input_ids'] = example['ids'][:-1]
    example['target_ids'] = example['ids'][1:]
    return example


def get_token(example):
    valid_lens = len(example['text'].split()) + 1
    example['valid_lens'] = valid_lens
    example['ids'] = tokenizer.encode(example['text']).ids
    example['input_ids'] = example['ids'][:-1]
    example['target_ids'] = example['ids'][1:]
    return example


def my_collate(batch):
    """for torch.DataLoader"""
    inputs = torch.tensor([data['input_ids'] for data in batch])
    targets = torch.tensor([data['target_ids'] for data in batch])
    valid_lens = torch.tensor([data['valid_lens'] for data in batch])
    return (inputs, targets, valid_lens)


if __name__ == '__main__':
    datasets = datasets.map(get_token, num_proc=4)
    datasets = datasets.remove_columns(['ids', 'text'])

    test_datasets = test_datasets.map(get_token, num_proc=4)
    test_datasets = test_datasets.remove_columns(['ids', 'text'])

    train_iter = DataLoader(datasets, batch_size=32, shuffle=True, collate_fn=my_collate)
    test_iter = DataLoader(test_datasets, batch_size=32, shuffle=False, collate_fn=my_collate)

    with open('train_iter.pkl', 'wb') as f:
        pickle.dump((train_iter, test_iter), f)
