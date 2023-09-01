import torch
from torch import nn
from d2l import torch as d2l


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

    def fit(self, train_iter: torch.tensor, test_iter: torch.tensor, max_epochs, device=d2l.try_gpu()):
        metric = d2l.Accumulator(4)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', legend=['train', 'test'])
        self.net = self.net.to(device)
        for epoch in range(max_epochs):
            num_iter = 1
            for X, Y, valid_lens_train in train_iter:
                X = X.to(device)
                Y = Y.to(device)
                valid_lens_train = valid_lens_train.to(device)
                output = self.net(X, valid_lens_train)
                loss = self.loss(output.permute(0, 2, 1), Y).sum()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.net.parameters(), 1.0)
                self.optimizer.step()

                # 测试部分
                if test_iter is not None:
                    accuracy = evaluate_accuracy(self.net, test_iter, device=device)
                else:
                    accuracy = 0
                metric.add(len(X), loss.item(), accuracy, 1)
                print('| epoch %d | iter %d/%d | loss %.4f | accuracy %.3f |' % (epoch + 1, num_iter,
                                                                 len(train_iter), metric[1]/metric[0], accuracy))
                num_iter += 1
            animator.add(epoch+1, [metric[1] / metric[0], metric[2] / metric[3]])


