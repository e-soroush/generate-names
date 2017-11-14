from models import RnnGenerative
from utils import english_letters, word2Variable, makeTargetVariable
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from datasets import startup_names, yield_dataset

n_length = len(english_letters) + 1
embedding_dim = 100
hidden_size = 150
batch_size = 32
use_gpu = torch.cuda.is_available()
generative = RnnGenerative(n_length, embedding_dim, hidden_size, batch_size)
if use_gpu:
    generative.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(generative.parameters(), lr=1e-3)
num_epochs = 5
names = startup_names()
generative.train()
for epoch in range(1, num_epochs+1):
    iterators = yield_dataset(names)
    for data in iterators:
        input_words = word2Variable(data)
        target_words = makeTargetVariable(data)
        if use_gpu:
            input_words = input_words.cuda()
            target_words = target_words.cuda()
        generative.zero_grad()
        output = generative(input_words)
        # print(output.size(), target_words.size())
        loss = 0
        for i in range(output.size()[0]):
            loss += criterion(output[i], target_words[i])
        loss.backward(retain_graph=True)
        optimizer.step()
        print('loss: {}'.format(loss.data.numpy()[0]))
    torch.save(generative.state_dict(), 'generative.m')

