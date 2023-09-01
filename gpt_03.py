from tokenizers import Tokenizer
import torch
from torch import nn
from gpt_02 import GPT, TrainGPT
from gpt_01_unuse import my_collate
import pickle
from d2l import torch as d2l

with open('train_iter.pkl', 'rb') as f:
    (train_iter, test_iter) = pickle.load(f)

tokenizer = Tokenizer.from_file('token.json')

num_hiddens = 200
num_heads = 10
num_layers = 4

net = GPT(tokenizer.get_vocab_size(), num_hiddens, num_heads,
          norm_shape=[30, num_hiddens], num_layers=num_layers, dropout=0.1)
# net.load_state_dict(torch.load('params.pkl'))


optimizer = torch.optim.Adam
loss = nn.CrossEntropyLoss()

trainer = TrainGPT(net, optimizer, loss, lr=0.0001)
trainer.fit(train_iter, None, max_epochs=2)
torch.save(net.state_dict(), 'params.pkl')

'''
# 预测部分
net.load_state_dict(torch.load('params.pkl'))
net = net.to(d2l.try_gpu())
net.eval()


input = 'waiting for your reply and wish you have a'
input_ids = torch.tensor([tokenizer.encode(input).ids])
valid_lens = torch.tensor([len(input.split())])
input_ids, valid_lens = input_ids.to(d2l.try_gpu()), valid_lens.to(d2l.try_gpu())
print(input_ids)
print(valid_lens)
output = torch.argmax(net(input_ids, valid_lens), dim=-1).squeeze(0).tolist()
print(output)
print(tokenizer.decode(output))
'''
