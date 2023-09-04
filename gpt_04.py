from tokenizers import Tokenizer
import torch
from gpt_02 import GPT, predict
from d2l import torch as d2l



# 超参数
num_hiddens = 300
num_heads = 10
num_layers = 4


tokenizer = Tokenizer.from_file('token.json')
net = GPT(tokenizer.get_vocab_size(), num_hiddens, num_heads,
          norm_shape=[30, num_hiddens], num_layers=num_layers, dropout=0.1)
net.load_state_dict(torch.load('params.pkl'))
net = net.to(d2l.try_gpu())
# net.eval()


print(tokenizer.decode([15]))
input = 'he is my best friend, so you need'
print(predict(input, net, tokenizer))
print(net.pos_embedding.data[0])