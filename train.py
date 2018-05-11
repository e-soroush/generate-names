from models import RnnGenerative
from utils import char2idx,save_checkpoint, create_train_state, load_checkpoint,read_text_file
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
from argparse import ArgumentParser


def train(starting_epoch, num_epochs, model, train_data_loader, batch_size, optimizer, names):
    criterion = nn.CrossEntropyLoss()
    try:
        for epoch in range(starting_epoch, num_epochs+1):
            model.train()
            pbar=tqdm(train_data_loader)
            for i,(x,y) in enumerate(pbar):
                model.zero_grad()
                x=Variable(x)
                y=Variable(y)
                if use_gpu:
                    x=x.cuda()
                    y=y.cuda()
                batch_size,word_max_len=x.size()
                model.init_hidden(batch_size)
                loss=0
                for i in range(word_max_len):
                    output=model(x[:,i].unsqueeze(1))
                    loss+=criterion(output,y[:,i])
                loss.backward(retain_graph=True)
                optimizer.step()
                pbar.set_description("Epoch {}/{} train loss: {}".format(epoch,num_epochs,loss.data[0]/word_max_len))

            save_checkpoint(create_train_state(model, epoch, optimizer), model_filename)
            generate_name(model, names)
    except KeyboardInterrupt:
        save_checkpoint(create_train_state(model, epoch-1, optimizer), model_filename)



if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('name_path',help="Path of name dataset. It is assumed names are included in one text file in a single line")
    parser.add_argument('--nb_epochs', help="Number epochs for training", default=100)
    parser.add_argument('--model_train_path', help="Path of pytorch model for resume training or storing model", default="model.pth.tar")
    parser.add_argument('--batch_size', help="Batch size", default=32)
    parser.add_argument('--embedding_dim', help="Size of embedding chars to a vector", default=50)
    parser.add_argument('--hidden_dim', help="Size of hidden dim for the used RNN network", default=50)
    parser.add_argument('--rnn_unit_type', help="Type of used rnn: LSTM, GRU, RNN", default='LSTM')
    parser.add_argument('--layer_num', help="Number of layers for RNN model", default=1)
    parser.add_argument('--embedding_dropout', help="Dropout for embedding", default=0.1)
    parser.add_argument('--final_dropout', help="Final dropout", default=0.3)
    parser.add_argument('--rnn_dropout', help="Rnn dropout", default=0)
    parser.add_argument('--use_gpu', help="If use gpu, considering it's available", default=True)
    args=parser.parse_args()

    nb_chars = len(char2idx)
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    use_gpu = torch.cuda.is_available() and args.use_gpu
    model = RnnGenerative(nb_chars, embedding_dim, hidden_dim, unit=args.rnn_unit_type, layer_num=args.layer_num,
                                embedding_dropout=args.embedding_dropout, final_dropout=args.final_dropout, rnn_dropout=args.rnn_dropout)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if use_gpu:
        model.cuda()
    model.init_hidden(batch_size)

    
    num_epochs = args.nb_epochs
    model_filename = args.model_train_path
    if not os.path.exists(args.name_path):
        raise FileNotFoundError("Cound not found %s"%args.name_path)
    names = read_text_file(args.name_path)
    train_data_loader=DataLoader(PeopleNames(names,chached=True),batch_size=batch_size,shuffle=True)
    starting_epoch = 1
    if os.path.exists(model_filename):
        model,optimizer,starting_epoch,_ = load_checkpoint(model_filename, model, optimizer)
    print('starting from epoch %d'%starting_epoch)
    train(starting_epoch,num_epochs,model,train_data_loader,batch_size,optimizer,names)

