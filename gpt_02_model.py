import torch
from torch import nn
from d2l import torch as d2l
from tokenizers import Tokenizer


def get_key_padding_mask(X, valid_lens):
    """
    得到key的padding_mask，所形成的mask用于计算得分score
    :param X: batch_size, sequence_size, hidden_size
    :param valid_lens: batch_size
    :return: type:bool
    """
    max_len = X.shape[1]
    mask = torch.arange(max_len, device=d2l.try_gpu())[None, :] < valid_lens[:, None]
    return ~mask


def get_attention_mask(key, querry):
    """

    :param key:
    :param querry:
    :return:
    """
    S = key.shape[1]
    L = querry.shape[1]
    mask = torch.arange(L, device=d2l.try_gpu())[:, None] < torch.arange(S, device=d2l.try_gpu())[None, :]
    return mask


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[-1] > 1:
        y_hat = d2l.argmax(y_hat, axis=-1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter, device):
    """Compute the accuracy for a model on a dataset.

        Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = d2l.Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, Y, valid_lens_test in data_iter:
            valid_lens_test = valid_lens_test.to(device)
            X = X.to(device)
            Y = Y.to(device)
            metric.add(accuracy(net(X, valid_lens_test), Y), d2l.size(Y))
    return metric[0] / metric[1]


class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, norm_shape, dropout=0):
        """GPT的解码器块"""
        super().__init__()
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout, batch_first=True)
        self.addnorm1 = d2l.AddNorm(norm_shape, dropout)
        self.ffn = d2l.PositionWiseFFN(num_hiddens, 2*num_hiddens, num_hiddens)
        self.addnorm2 = d2l.AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 计算padding_mask
        key_padding_mask = get_key_padding_mask(X, valid_lens)
        # 计算attention_mask
        attention_mask = get_attention_mask(X, X)
        attn_output, attn_weights = self.attention(X, X, X,
                                                   key_padding_mask=key_padding_mask, attn_mask=attention_mask)
        X1 = self.addnorm1(X, attn_output)
        ffn_output = self.ffn(X1)
        Y = self.addnorm2(X1, ffn_output)
        return Y, attn_weights


class GPT(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_heads, norm_shape, num_layers, dropout=0, max_len=30):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'blk{i}', DecoderBlock(num_hiddens, num_heads, norm_shape, dropout))
        self.output = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, valid_lens):
        X = self.text_embedding(X) + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X, atten_weights = blk(X, valid_lens)
        output = self.output(X)
        return output


class TrainGPT(nn.Module):
    def __init__(self, net, optimizer, loss, lr):
        super().__init__()
        self.net = net
        self.lr = lr
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.loss = loss

    def fit(self, train_iter: torch.tensor, test_iter: torch.tensor, max_epochs, device=d2l.try_gpu(),
            is_funtunning=False):
        metric = d2l.Accumulator(4)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', legend=['train', 'test'])
        self.net = self.net.to(device)
        for epoch in range(max_epochs):
            num_iter = 1
            for X, Y, valid_lens_train in train_iter:
                X = X.to(device)
                Y = Y.to(device)
                print(valid_lens_train)
                valid_lens_train = valid_lens_train.to(device)
                output = self.net(X, valid_lens_train)
                # 判断是否为微调
                if is_funtunning is not True:
                    loss = self.loss(output.permute(0, 2, 1), Y).sum()
                else:
                    loss = self.loss(output, Y).sum()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()

                # 测试部分
                if test_iter is not None:
                    accuracy = evaluate_accuracy(self.net, test_iter, device=device)
                else:
                    accuracy = 0
                metric.add(len(X), loss.item(), accuracy, 1)
                print('| epoch %d | iter %d/%d | loss %.4f | accuracy %.3f |' % (epoch + 1, num_iter,
                                                                 len(train_iter), metric[1]/metric[0], accuracy))
                # 保存参数
                if num_iter % 100 == 0:
                    torch.save(self.net.state_dict(), "params.pkl")

                num_iter += 1
            animator.add(epoch+1, [metric[1] / metric[0], metric[2] / metric[3]])


def predict(input: str, net: GPT, tokenizer: Tokenizer, device=d2l.try_gpu()):

    input_ids = torch.tensor([tokenizer.encode(input).ids], device=device)[:, :-1]
    valid_lens = torch.tensor([len(input.split()) + 2], device=device)
    output = net(input_ids, valid_lens)
    print(output[0, -1, :])
    output = torch.argmax(output, dim=-1).squeeze(0)
    output_token = tokenizer.decode(output.tolist()).split()

    return output_token


class GPTClassify(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_heads, norm_shape, num_layers, dropout=0, max_len=30, num_features=2):
        super().__init__()
        self.GPT = GPT(vocab_size, num_hiddens, num_heads, norm_shape, num_layers, dropout=dropout, max_len=max_len)
        self.linear = nn.Linear(vocab_size, num_features)

    def forward(self, X, valid_lens):
        output = self.GPT(X, valid_lens)[:, -1, :]
        # output.shape:batch_size, 1, vocab_size
        output = self.linear(output).squeeze(1)
        # output.shape:batch_size, num_features
        return output
