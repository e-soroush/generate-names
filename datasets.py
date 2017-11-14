import pandas as pd
from utils import unicodeToAscii, shuffle_data

def startup_names():
    dataset = pd.read_csv('/home/esoroush/Datasets/startups.csv')
    dataset = dataset.dropna(how='any')
    names=[]
    for i, item in dataset.iterrows():
        if isinstance(item['name'], str):
            name = unicodeToAscii(item['name'].decode('utf8'))
            if len(name) > 0:
                names += [name]
    return names

def yield_dataset(X, y=None, batchsize=32, shuffle=True):
    if y is not None:
        assert(len(X) == len(y))
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete batches are submitted
    for i in range(len(X)//batchsize):
        if y is not None:
            yield X[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]
        else:
            yield X[i*batchsize:(i+1)*batchsize]