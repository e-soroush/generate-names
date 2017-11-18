from models import RnnGenerative, english_letters
from params import *
from utils import char_tensor, load_checkpoint
import torch

n_length = len(english_letters) + 1
embedding_dim = 100
hidden_size = 150
max_length = 10
model = RnnGenerative(len(english_letters) + 1, embedding_dim, hidden_size, unit='lstm', layer_num=2)
model = load_checkpoint('names-lstm-layer-2.model', model)
first_word = 'E'
word = []
model.init_hidden(1)
model.eval()
tempretature = 0.5
for i in range(max_length):
    word += [first_word]
    input = char_tensor(first_word)
    output = model(input)
    char_weights = output.squeeze().data.div(tempretature).exp().cpu()
    char_idx = torch.multinomial(char_weights, 1)[0]
    # char_index = output.data.max(dim=1)[1].numpy()[0]
    if char_idx == len(english_letters):
        print('word: %s'%''.join(word))
        break
    first_word = english_letters[char_idx]
print('Word: %s'%''.join(word))

