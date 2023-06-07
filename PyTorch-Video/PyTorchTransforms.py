import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform):
        #load data
        xy = np.loadtxt('C:\\Users\\Vinson\\.vscode\\projects\\python\\pytorch\\data\\wine\\wine.csv', delimiter = ",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:,[0]]
        
        #apply transform if available
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
    
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    def __len__(self):
        return self.n_samples
    

class ToTensor: #class to convert numpy to tensor
    def __call__(self, sample):
        inputs, targets = sample    
        return torch.from_numpy(inputs,), torch.from_numpy(targets)
    
class MulTransform: #class to multiply 
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


#example
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

#composed is a transform combined with two or more transforms
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))