import random
from pathlib import Path


def split_data(data_dir, train_size=0.8, val_size=0.1):
    random.seed(1234)
    data = Path('data').glob('*/*')
    data = [x for x in data if x.is_file() and x.suffix != '.zip']
    random.shuffle(data)
    train_size = int(len(data) * train_size)
    val_size = int(len(data) * val_size)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    return train_data, val_data, test_data
