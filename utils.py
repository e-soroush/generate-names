import unicodedata
import string
import torch
import numpy as np
import pandas as pd
import glob2
import numpy as np

english_letters = string.ascii_letters[:26]
# add start end and padding tokens
START_TOKEN='<s>'
END_TOKEN='<e>'
PADDING_TOKEN='<p>'
UNKOWN_TOKEN='<unk>'
char2idx=[PADDING_TOKEN,UNKOWN_TOKEN,START_TOKEN,END_TOKEN,' ']+list(english_letters)

def tokenize_words(word):
    """
    Tokenize each character of words
    """
    if not isinstance(word,list):
        word=[word]
    max_word_len=len(max(word,key=lambda f:len(f)))+2
    tokenized=np.zeros((len(word),max_word_len),np.int32)
    for i,w in enumerate(word):
        for j,c in enumerate(w.lower()):
            try:
                tokenized[i,j+1]=char2idx.index(c)
            except:
                tokenized[i,j+1]=char2idx.index(UNKOWN_TOKEN)
        tokenized[i,0]=char2idx.index(START_TOKEN)
        tokenized[i,j+2]=char2idx.index(END_TOKEN)
    return tokenized


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in english_letters
    )


def read_from_csv(path,name_field='name'):
    dataset = pd.read_csv(path)
    dataset = dataset.dropna(how='any')
    names=[]
    for i, item in dataset.iterrows():
        if isinstance(item[name_field], str):
            name = unicodeToAscii(item[name_field])
            if len(name) > 0:
                names += [name]
    return names


def read_people_names(glob_pattern='/home/esoroush/Datasets/names/*.txt'):
    filenames = glob2.glob(glob_pattern)
    names = []
    for filename in filenames:
        with open(filename, 'r') as fhandler:
            for name in fhandler:
                names += [name.strip().split(',')[0].lower()]
    return names

def read_text_file(path):
    with open(path) as fhandler:
        names=[f.strip() for f in fhandler]
    return names

def save_checkpoint(model, filename, is_best=False):
    torch.save(model, filename)
    if is_best:
        torch.save(model, 'best_'+filename)

def create_train_state(model, epoch, optimizer, best_precision=0.0):
    return {'epoch': epoch+1, 
            'state_dict':model.state_dict(),
            'best_precision':best_precision,
            'optimizer':optimizer.state_dict()}

def load_checkpoint(model_filename, model, optimizer=None):
    structure = torch.load(model_filename)
    model.load_state_dict(structure['state_dict'])
    if optimizer:
        optimizer.load_state_dict(structure['optimizer'])
        return model, optimizer, structure['epoch'], structure['best_precision']
    return model