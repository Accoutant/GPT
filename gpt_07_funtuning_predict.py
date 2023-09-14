from gpt_02_model import GPTClassify, predict_funtuning
import torch
from tokenizers import Tokenizer
import pickle

with open("fine_tunning.pkl", 'rb') as f:
    tokenizer, tran_iter, test_iter = pickle.load(f)

# 定义超参数
num_hiddens = 300
num_heads = 10
num_layers = 4
# 定义神经网络
net = GPTClassify(tokenizer.get_vocab_size(), num_hiddens, num_heads,
                  norm_shape=[30, num_hiddens], num_layers=num_layers, dropout=0.1, num_features=3)

net.load_state_dict(torch.load("params.pkl"))

text = "you are my best friend"
predict_funtuning(net, text, tokenizer)