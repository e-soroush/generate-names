from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
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

class RandomSampler(Sampler):
    def __init__(self, nb_samples, desired_samples, shuffle=True):
        self.data_samples = nb_samples
        self.desired_samples = desired_samples
        self.shuffle=shuffle

    def gen_sample_array(self):
        n_repeats = self.desired_samples // self.data_samples
        self.sample_idx_array = torch.range(0,self.data_samples-1).long()
        if n_repeats>0:
            self.sample_idx_array=self.sample_idx_array.repeat(n_repeats)
        if self.shuffle:
          self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))][:self.desired_samples]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.desired_samples