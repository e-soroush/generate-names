import unicodedata
import string
import torch
import numpy as np

english_letters = string.ascii_letters + ".,;'-"


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in english_letters
    )

# One-hot matrix of first to last letters (not including EOS) for input
def word2Variable(words, max_length=20):
    if not isinstance(words, list):
        raise('You have to enter words in list format')
    tensor = torch.zeros(max_length, len(words)).fill_(len(english_letters)).type(torch.LongTensor)
    for i, word in enumerate(words):
        if len(word) >= max_length:
            word = word[:max_length-1]
        for li in range(len(word)):
            letter = word[li]
            tensor[li][i] = english_letters.find(letter)
    return torch.autograd.Variable(tensor)

# LongTensor of second letter to end (EOS) for target
def makeTargetVariable(words, max_length=20):
    if not isinstance(words, list):
        raise('You have to enter words in list format')
    letter_indexes = torch.zeros(max_length, len(words)).fill_(len(english_letters)).type(torch.LongTensor)
    for i, word in enumerate(words):
        if len(word) >= max_length:
            word = word[:max_length-1]
        for li in range(1, len(word)):
            letter = word[li]
            letter_indexes[li][i] = english_letters.find(letter)
    return torch.autograd.Variable(letter_indexes)


def shuffle_data(X, y=None):
    if isinstance(X, list):
        X = np.array(X)
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    X = X.tolist()
    if y is not None:
        if isinstance(y, list):
            y = np.array(y)
        y = y[s]
        y = y.tolist()
    return X, y
