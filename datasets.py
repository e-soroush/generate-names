from torch.utils.data import Dataset
import torch
from utils import tokenize_words
from torch.autograd import Variable
import pickle
import os

class PeopleNames(Dataset):
    def __init__(self, name_list, chached=True):
        super(PeopleNames,self).__init__()
        cache_path='.peopleNames.p'
        if chached and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fh:
                self.tokenized=pickle.load(fh)
        else:
            self.tokenized=(torch.from_numpy(tokenize_words(name_list))).long()
            with open(cache_path, 'wb') as fh:
                pickle.dump(self.tokenized,fh)
    

    def __getitem__(self, index):
        word=self.tokenized[index]
        target=word[1:]
        word=word[:-1]
        return word,target
    def __len__(self):
        return len(self.tokenized)


def yield_dataset(X, y=None, batch_size=32, shuffle=True):
    if y is not None:
        assert(len(X) == len(y))
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete batches are submitted
    for i in range(len(X)//batch_size):
        if y is not None:
            yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
        else:
            yield X[i*batch_size:(i+1)*batch_size]
    
