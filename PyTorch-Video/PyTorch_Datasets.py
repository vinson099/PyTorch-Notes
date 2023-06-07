import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #load data
        xy = np.loadtxt('C:\\Users\\Vinson\\.vscode\\projects\\python\\pytorch\\data\\wine\\wine.csv', delimiter = ",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:,[0]]) #n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
batch_size = 4
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True, num_workers=0)

#training loop
numEpochs = 2
totalSamples = len(dataset)
nIterations = math.ceil(totalSamples/batch_size)
# print(totalSamples, nIterations)

for epoch in range(numEpochs):
    for i, (inputs, labels) in enumerate(dataloader): #enumerate gives us index of loop
        #forward backward, update not included
        if(i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{numEpochs}, step {i+1}/{nIterations}, inputs{inputs.shape}')
