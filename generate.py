from models import RnnGenerative
from utils import load_checkpoint,char2idx, START_TOKEN,END_TOKEN, read_text_file
import torch
import os
from torch.autograd import Variable
from argparse import ArgumentParser

def generate_name(model, names):
    max_length = 17
    model.init_hidden(1)
    model.eval()
    tempretature = 0.9
    first_word = Variable(torch.LongTensor([char2idx.index(START_TOKEN)])).cuda()
    word=[]
    for i in range(max_length):
        output = model(first_word.unsqueeze(1))
        char_weights = output.squeeze().data.div(tempretature).exp().cpu()
        char_idx = torch.multinomial(char_weights[3:], 1)[0]
        if char_idx == 0:
            print('end token!')
            break
        first_word=Variable(first_word.data.new([char_idx+3]))
        next_char = char2idx[char_idx+3]
        word.append(next_char)
    word = ''.join(word)
    print('Word: %s'%word)
    try:
        print('Found %s'%names[names.index(word)])
    except:
        print('The word %s is not in train names'%word)

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--name_path',help="Path of name dataset. It is assumed names are included in one text file in a single line", default="")
    parser.add_argument('--nb_epochs', help="Number epochs for training", default=100)
    parser.add_argument('--model_train_path', help="Path of pytorch model for resume training or storing model", default="model.pth.tar")
    parser.add_argument('--embedding_dim', help="Size of embedding chars to a vector", default=50)
    parser.add_argument('--hidden_dim', help="Size of hidden dim for the used RNN network", default=50)
    parser.add_argument('--rnn_unit_type', help="Type of used rnn: LSTM, GRU, RNN", default='LSTM')
    parser.add_argument('--layer_num', help="Number of layers for RNN model", default=1)
    parser.add_argument('--embedding_dropout', help="Dropout for embedding", default=0.1)
    parser.add_argument('--final_dropout', help="Final dropout", default=0.3)
    parser.add_argument('--rnn_dropout', help="Rnn dropout", default=0)
    parser.add_argument('--use_gpu', help="If use gpu, considering it's available", default=True)
    args=parser.parse_args()

    model_filename = args.model_train_path
    names=[]
    if os.path.exists(args.name_path):
        names = read_text_file(args.name_path)
    nb_chars = len(char2idx)
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    use_gpu = torch.cuda.is_available() and args.use_gpu
    model = RnnGenerative(nb_chars, embedding_dim, hidden_dim, unit=args.rnn_unit_type, layer_num=args.layer_num,
                                embedding_dropout=args.embedding_dropout, final_dropout=args.final_dropout, rnn_dropout=args.rnn_dropout)
    if use_gpu:
        model.cuda()
    model.init_hidden(1)
    if os.path.exists(model_filename):
        model= load_checkpoint(model_filename, model)
    generate_name(model, names)