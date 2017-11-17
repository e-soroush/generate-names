import torch
from torch import nn
from utils import *

class RnnGenerative(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, layer_num=1):
        super(RnnGenerative, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax()
        self.hidden = self.init_hidden()
    def init_hidden(self, batch_size=32):
        return torch.autograd.Variable(torch.randn(self.layer_num, batch_size, self.hidden_size))
    def forward(self, x_input):
        batch_size = x_input.size(0)
        x = self.embedding(x_input)
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
        out = self.softmax(self.out(out.view(batch_size, -1)))
        return out



