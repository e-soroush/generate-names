from models import RnnGenerative, char2idx,START_TOKEN,END_TOKEN
from params import *
from utils import char_tensor, load_checkpoint, people_names
import torch
from torch.autograd import Variable

names = people_names()
def generate_name(model):
    max_length = 17
    model.init_hidden(1)
    model.eval()
    tempretature = 0.5
    first_word = Variable(torch.LongTensor([char2idx.index(START_TOKEN)])).cuda()
    word=[]
    for i in range(max_length):
        output = model(first_word)
        char_weights = output.squeeze().data.div(tempretature).exp().cpu()
        char_idx = torch.multinomial(char_weights[3:], 1)[0]
        # _, char_idx=output.max(0)
        # char_idx=char_idx.data[0]
        # char_index = output.data.max(dim=1)[1].numpy()[0]
        if char_idx == 0:
            print('end token!')
            break
        next_char = char2idx[char_idx+3]
        word.append(next_char)
    word = ''.join(word)
    print('Word: %s'%word)
    try:
        print('Found %s'%names[names.index(word)])
    except:
        print('The word %s is not in train names'%word)

if __name__=='__main__':
    n_length = len(char2idx)
    embedding_dim = 50
    hidden_size = 150
    model = RnnGenerative(n_length, embedding_dim, hidden_size, unit='lstm', layer_num=1)
    model = load_checkpoint('people-lstm-layer-2.model', model)
    model.cuda()
    generate_name(model)