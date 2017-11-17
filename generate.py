from models import RnnGenerative, english_letters
from params import *
from utils import char_tensor
import torch

n_length = len(english_letters) + 1
embedding_dim = 100
hidden_size = 150
max_length = 10
model = RnnGenerative(len(english_letters) + 1, embedding_dim, hidden_size)
model.load_state_dict(torch.load('generative_names.model'))
first_word = 'E'
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

