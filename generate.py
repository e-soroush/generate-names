from models import RnnGenerative, english_letters
from params import *
from utils import char_tensor, load_checkpoint
import torch

n_length = len(english_letters) + 1
embedding_dim = 100
hidden_size = 150
max_length = 10
model = RnnGenerative(len(english_letters) + 1, embedding_dim, hidden_size)
model = load_checkpoint('names.model', model)
first_word = 'S'
word = []
model.hidden = model.init_hidden(1)
for i in range(max_length):
    word += [first_word]
    input = char_tensor(first_word)
    output = model(input)
    char_index = output.data.max(dim=1)[1].numpy()[0]
    print(char_index)
    if char_index == len(english_letters):
        print('word: %s'%''.join(word))
        break
    first_word = english_letters[char_index]
print('Word: %s'%''.join(word))

