from models import RnnGenerative
from utils import char2idx, word2Variable, makeTargetVariable, save_checkpoint, create_train_state, load_checkpoint,people_names
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from datasets import PeopleNames
from tqdm import tqdm
import os
from torch.utils.data.dataloader import DataLoader
from generate import generate_name
import numpy as np

nb_chars = len(char2idx)
print('nb_chars %d'%nb_chars)
embedding_dim = 50
hidden_size = 150
batch_size = 32
use_gpu = torch.cuda.is_available()
generative = RnnGenerative(nb_chars, embedding_dim, hidden_size, unit='lstm', layer_num=1,
                            embedding_dropout=0, final_dropout=0.1, rnn_dropout=0)
if use_gpu:
    generative.cuda()
generative.init_hidden(batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(generative.parameters(), lr=1e-3)
num_epochs = 500
model_filename = 'weights.pth'
names = people_names()
np.random.seed(11)
np.random.shuffle(names)
train_data_loader=DataLoader(PeopleNames(names,chached=True),batch_size=batch_size,shuffle=True)
starting_epoch = 1
if os.path.exists(model_filename):
    generative,optimizer,starting_epoch,_ = load_checkpoint(model_filename, generative, optimizer)
    
print('starting from epoch %d'%starting_epoch)
try:
    for epoch in range(starting_epoch, num_epochs+1):
        generative.train()
        pbar=tqdm(train_data_loader)
        for i,(x,y) in enumerate(pbar):
            generative.zero_grad()
            x=Variable(x)
            y=Variable(y)
            if use_gpu:
                x=x.cuda()
                y=y.cuda()
            batch_size,word_max_len=x.size()
            generative.init_hidden(batch_size)
            loss=0
            for i in range(word_max_len):
                output=generative(x[:,i].unsqueeze(1))
                loss+=criterion(output,y[:,i])
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_description("Epoch {}/{} train loss: {}".format(epoch,num_epochs,loss.data[0]/word_max_len))

        save_checkpoint(create_train_state(generative, epoch, optimizer), model_filename)
        generate_name(generative)
except KeyboardInterrupt:
    save_checkpoint(create_train_state(generative, epoch-1, optimizer), model_filename)
