import torch
from torch.utils.data import TensorDataset, DataLoader

class WrappedDataLoader:
    
    def __transform(self, x, func):
        x_transformed = torch.tensor(x)
        if func:
            x_transformed = func(x_transformed)
            
        return x_transformed
    
    def __init__(self, x, y, x_trans=None, y_trans=None, **kwargs):
        x_transformed = self.__transform(x, x_trans)
        y_transformed = self.__transform(y, y_trans)
        self.dl = DataLoader(TensorDataset(x_transformed, y_transformed), **kwargs)
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        return iter(self.dl)

def x_transform(x):
    return x.view(-1, 1, 28, 28)

def y_transform(y):
    pass
