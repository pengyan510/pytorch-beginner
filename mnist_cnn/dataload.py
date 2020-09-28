import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import trans_func

class MNISTDataset(Dataset):

    def __init__(self, x, y, trans_func=None):
        self.x = x
        self.y = y
        self.trans_func = trans_func

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {'x': self.x[idx], 'y': self.y[idx]}
        if self.trans_func:
            sample = self.trans_func(sample)

        return sample


def createDataLoader(x, y, **kwargs):
    dataset = MNISTDataset(x, y, trans_func)
    return DataLoader(dataset, **kwargs)
