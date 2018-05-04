from models import RnnGenerative, char2idx,START_TOKEN,END_TOKEN
from params import *
from utils import char_tensor, load_checkpoint, people_names
import torch
from torch.autograd import Variable

n_length = len(char2idx)
embedding_dim = 128
hidden_size = 150
max_length = 17
model = RnnGenerative(n_length, embedding_dim, hidden_size, unit='lstm', layer_num=2)
model = load_checkpoint('people-lstm-layer-2.model', model)
model.init_hidden(1)
model.eval()
tempretature = 0.8
names = people_names()
first_word = Variable(torch.LongTensor([char2idx.index(START_TOKEN)]))
word=[]
for i in range(max_length):
    output = model(first_word)
    _, char_idx=output.max(0)
    char_idx=char_idx.data[0]
    # char_weights = output.squeeze().data.div(tempretature).exp().cpu()
    # char_idx = torch.multinomial(char_weights, 1)[0]
    # char_index = output.data.max(dim=1)[1].numpy()[0]
    if char_idx == 2:
        break
    next_char = char2idx[char_idx]
    word.append(next_char)
word = ''.join(word)
print('Word: %s'%word)
try:
    print('Found %s'%names[names.index(word)])
except:
    print('The word %s is not in train names'%word)
