from tokenizers import Tokenizer
import torch
from torch import nn
from gpt_02_model import GPT, TrainGPT
from gpt_01_process_data import my_collate
import pickle


with open('train_iter.pkl', 'rb') as f:
    (train_iter, test_iter) = pickle.load(f)

tokenizer = Tokenizer.from_file('token.json')

num_hiddens = 300
num_heads = 10
num_layers = 4

net = GPT(tokenizer.get_vocab_size(), num_hiddens, num_heads,
          norm_shape=[30, num_hiddens], num_layers=num_layers, dropout=0.1)
# net.load_state_dict(torch.load('params.pkl'))


optimizer = torch.optim.Adam
loss = nn.CrossEntropyLoss()

trainer = TrainGPT(net, optimizer, loss, lr=0.0001)
trainer.fit(train_iter, test_iter, max_epochs=2)
torch.save(net.state_dict(), 'params.pkl')

