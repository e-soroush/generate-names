import torch
from torch import nn
from utils import *

class RnnGenerative(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, dropout=0.5, layer_num=1, unit='rnn'):
        super(RnnGenerative, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.unit = unit
        self.embedding = nn.Embedding(input_size, embedding_dim)
        if self.unit == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_size, layer_num)
        elif self.unit == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, layer_num)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(dropout)
        self.init_hidden()

    def init_hidden(self, batch_size=32, cuda=False):
        if self.unit == 'rnn':
            self.hidden=torch.autograd.Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size))
            if cuda:
                self.hidden = self.hidden.cuda()
        elif self.unit == 'lstm':
            self.hidden=torch.autograd.Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)),torch.autograd.Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size))
            if cuda:
                self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
    
    def forward(self, x_input):
        batch_size = x_input.size(0)
        x = self.embedding(x_input)
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
        out = self.dropout(out)
        out = self.softmax(self.out(out.view(batch_size, -1)))
        return out



