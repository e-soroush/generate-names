import torch
from torch import nn
from torch.autograd import Variable
from utils import *

rnn_modules=['RNN','LSTM','GRU']
class RnnGenerative(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size,
                 embedding_dropout=0.1,rnn_dropout=0.5,final_dropout=0.5,
                 layer_num=1, unit='rnn'):
        super(RnnGenerative, self).__init__()
        unit=unit.upper()
        if unit not in rnn_modules:
            raise ValueError("unit value should be in %s"%', '.join(rnn_modules))
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.unit = unit
        self.embedding = nn.Embedding(input_size, embedding_dim,padding_idx=0)
        self.rnn=getattr(nn,unit)(embedding_dim, hidden_size, layer_num, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)
        self.final_dropout = nn.Dropout(final_dropout)
        self.init_hidden()

    def init_hidden(self, batch_size=32):
        # just to make sure it's on the same device with model
        weights=next(self.parameters())
        hidden_shape=(self.layer_num*(1+self.rnn.bidirectional), batch_size, self.hidden_size)
        if self.unit.lower() == 'lstm':
            self.hidden=(Variable(weights.data.new(*hidden_shape).zero_()),Variable(weights.data.new(*hidden_shape).zero_()))
        else:
            self.hidden=Variable(weights.data.new(*hidden_shape).zero_())
    
    def forward(self, x_input):
        # make sure input data is batch first
        # batch_size x word_length
        embedded = self.embedding(x_input)
        out, self.hidden = self.rnn(embedded.unsqueeze(1), self.hidden)
        out = self.final_dropout(out)
        out = self.softmax(self.output_layer(out)).squeeze()
        return out



