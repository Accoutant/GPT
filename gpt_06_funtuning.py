from gpt_02_model import GPTClassify, TrainGPT
import torch
from torch import nn
from d2l import torch as d2l
import pickle
from gpt_01_process_data import my_collate
from tokenizers import Tokenizer

with open("fine_tunning.pkl", 'rb') as f:
    tokenizer, tran_iter, test_iter = pickle.load(f)


# 定义超参数
num_hiddens = 300
num_heads = 10
num_layers = 4
# 定义神经网络
net = GPTClassify(tokenizer.get_vocab_size(), num_hiddens, num_heads,
                  norm_shape=[30, num_hiddens], num_layers=num_layers, dropout=0.1, num_features=3)
net.GPT.load_state_dict(torch.load('params.pkl'))
net = net.to(d2l.try_gpu())
# 定义训练器
optimizer = torch.optim.Adam
loss = nn.CrossEntropyLoss()
trainer = TrainGPT(net, optimizer, loss, 1e-4)
trainer.fit(tran_iter, test_iter, max_epochs=2, is_funtunning=True)
