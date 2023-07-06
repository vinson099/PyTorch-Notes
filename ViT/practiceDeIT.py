import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
%matplotlib inline

import sys
from tqdm import tqdm
import time
import copy

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        # add filter/transformations to train dataset
        preprocess = transforms.Compose([
            #random flips
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #slightly change colors in img
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.25),
            #transformations for crop
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #noramlize tensors
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
            #randomly erase
            transforms.RandomErasing(p=0.2, value='random')
        ])
        #load train data
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = preprocess)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = transforms.Compose([ # We dont need augmentation for val/test transforms
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
        ])
        #load data
        try:
            val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
        except FileNotFoundError:
            val_data = datasets.ImageFolder(os.path.join(data_dir, "validate/"), transform=transform)
        except FileNotFoundError:
            val_data = datasets.ImageFolder(os.path.join(data_dir, "val/"), transform=transform)
        else:
            print("File not found")
        
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, test_loader, len(val_data), len(test_data)
    
#get path
# dataset from https://www.kaggle.com/datasets/gpiosenka/100-bird-species
dataset_path = "/home/vinso/vscode-projects/repos/ImageClassificationData/kaggle-birds"
#load data
(train_loader, train_data_len) = get_data_loaders(dataset_path, 128, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)

#get classes
classes = get_classes("/home/vinso/vscode-projects/repos/ImageClassificationData/kaggle-birds/train/")
print(classes, len(classes))

#get dataloaders and amount of data
dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}

print(len(train_loader), len(val_loader), len(test_loader))

print(train_data_len, valid_data_len, test_data_len)

#load model from checkpoint
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)


for param in model.parameters(): #freeze model
    #will not calculate grad
    param.requires_grad = False
    #model will only do feature extraction using pretrained weights


n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
)

model = model.to(device)
print(model.head)

'''
Sequential(
  (0): Linear(in_features=192, out_features=512, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.3, inplace=False)
  (3): Linear(in_features=512, out_features=50, bias=True)
)
'''
#set loss function and optimizer
criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
optimizer = optim.Adam(model.head.parameters(), lr=0.001)

# set lr scheduler
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

#training loop 
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    #calculate time elapsed
    since = time.time()
    #copy best model
    best_model_wts = copy.deepcopy(model.state_dict())
    #initialize best accuracy variable
    best_acc = 0.0
    
    #t
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']: # We do training and validation phase per epoch
            if phase == 'train':
                model.train() # model to training mode
            else:
                model.eval() # model to evaluate
            
            #initialize running variables
            running_loss = 0.0
            running_corrects = 0.0
            
            #iterates over data in train/val-loader[phase]
            #tqdm for progress bar
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #zero grad to reset past iterations
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # no autograd makes validation go faster
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # used for accuracy
                    loss = criterion(outputs, labels) # get loss from loss func
                    
                    if phase == 'train': 
                        loss.backward() #calculate loss
                        optimizer.step() #step optimizer
                #calculate total loss and total correct
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                #step LR scheduler
                scheduler.step() # step at end of epoch
            
            #calc average loss/acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        print()

    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer, lr_scheduler) # now it is a lot faster

#testing loop
#initialize running var
test_loss = 0.0
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
model.eval()

#loop through test_loader
#tqdm for progress bar
for data, target in tqdm(test_loader):
    data, target = data.to(device), target.to(device)


    with torch.no_grad(): # turn off autograd for faster testing
        #calculate loss and get output values
        output = model(data)
        loss = criterion(output, target)

    #calculate test_loss
    test_loss = loss.item() * data.size(0)
    _, pred = torch.max(output, 1) # get highest value for each img/get pred
    correct_tensor = pred.eq(target.data.view_as(pred)) # create boolean tensor to check if pred matches target data
    correct = np.squeeze(correct_tensor.cpu().numpy()) #creates numpy arr with 1/0 to represent if each pred was right/wrong
    if len(target) == 32:
        for i in range(32):
            #calculate amt correct
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

#avg test loss
test_loss = test_loss / test_data_len
print('Test Loss: {:.4f}'.format(test_loss))
for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
            classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test accuracy of %5s: NA" % (classes[i]))
print("Test Accuracy of %2d%% (%2d/%2d)" % (
            100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)
        ))

#reset test loss
test_loss = 0.0
#init class variables
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
model.eval()

# iterate through data in test_laoder 
# tqdm for progress bar
for data, target in tqdm(test_loader):
    #move to device
    data, target = data.to(device), target.to(device)
    with torch.no_grad(): # turn off autograd for faster testing
        #calc loss and get model outputs
        output = model(data)
        loss = criterion(output, target)
    #calc test loss
    test_loss = loss.item() * data.size(0)
    _, pred = torch.max(output, 1) #get highest val prediction
    correct_tensor = pred.eq(target.data.view_as(pred)) # create boolean tensor to check if pred matches target data
    correct = np.squeeze(correct_tensor.cpu().numpy()) #creates numpy arr with 1/0 to represent if each pred was right/wrong
    if len(target) == 32:
        for i in range(32):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
# avg test loss
test_loss = test_loss / test_data_len
print('Test Loss: {:.4f}'.format(test_loss))
for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
            classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test accuracy of %5s: NA" % (classes[i]))
print("Test Accuracy of %2d%% (%2d/%2d)" % (
            100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)
        ))


#save model
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model.cpu(), example)
traced_script_module.save("birds_deit.pt")