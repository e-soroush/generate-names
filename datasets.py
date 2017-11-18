import pandas as pd
from utils import unicodeToAscii, shuffle_data

def startup_names():
    dataset = pd.read_csv('/home/esoroush/Datasets/startups.csv')
    dataset = dataset.dropna(how='any')
    names=[]
    for i, item in dataset.iterrows():
        if isinstance(item['name'], str):
            name = unicodeToAscii(item['name'])
            if len(name) > 0:
                names += [name]
    return names

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