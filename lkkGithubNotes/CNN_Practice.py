import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#check gpu availability
has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 40
# percentage of training set to use as validation
valid_size = 0.1


#Helper Function
#un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize image
    plt.imshow(np.transpose(img, (1, 2, 0)))  
        # convert from Tensor image, permute the axes according to the values given


#Transform data into normalized tensor with values -1 or 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
        #does not need extra 0.5 in each parenthesis b/c there is only one color channel
    ])


#Import FashionMNIST Dataset 
    # 10 labels
    # training set of 60,000
    # test set of 10,000
train_data = datasets.FashionMNIST('data', train=True,
                                   download=True, transform = transform)
test_data = datasets.FashionMNIST('data', train=False,
                                   download=False, transform = transform)


#Split data into training set and validation set
num_train = len(train_data)                             # find number of samples in training data
indices = list(range(num_train))                        # create list of indices for number of samples
np.random.shuffle(indices)                              # shuffle data randomly
split = int(np.floor(valid_size * num_train))           # find the amount of training data to split dataset to use as validation
train_idx, valid_idx = indices[split:], indices[:split] # splits the data to be used for training and validation

#create samplers to obtain batches of data from dataset
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
    #use SubsetRandomSampler to access random elements from index 


# create data loaders to access dataset using 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


# create image labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dresses', 'Coat', 'Sandal', 
           'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#visualize the data
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(36, 10)) #25in wide and 4in tall
# display 40 images
for idx in np.arange(40):
    ax = fig.add_subplot(4, int(batch_size/4), idx+1, xticks=[], yticks=[]) #4 rows, 10 columns
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])


#print(images.shape)
#output -> (40, 1, 28, 28)
#Remove single-dimensional entries from the shape of an array
bw_img = np.squeeze(images)

#display data in black/white
fig = plt.figure(figsize = (36, 10)) 

for i in np.arange(bw_img.shape[0]):        #loops through 40 images
    plt.subplot(4, 10, i+1)                 #4 rows, 10 columns, index i+1
    plt.imshow(bw_img[i], cmap='gray')      #displays images in gray
    img = bw_img[i]                         #img is the individual image
    width, height = img.shape               #find height and width
    ax = fig.add_subplot(4, int(batch_size/4), idx+1, xticks=[], yticks=[]) #4 rows, 10 columns
    thresh = img.max()/2.5                  #find threshold value used to determine font color for annotation

    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')
            


#CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     #inherit qualities from nn.Module


        #convolutional layer(input, output, kernel size)

        #Convolutional layer 1: sees 28x28x1 image tensor
                #first input is color channel #
                #14 filters -> 14 output 
                #kernel size is the small filter panning across image
        self.conv1 = nn.Conv2d(1, 14, 3, padding=1) 
        #filters = size after conv = (28-3+(2*1))/1 + 1 = 28, output size: 28*28*14
            # 28-3 -> subtract kernel size from input dimension
            # +2*1 -> add 2(padding value)
            # /1   -> divide by stride 
            # +1   -> add padding pixel

        #pooling layer: 28*28*14 -> 14*14*14
            #divide by pooling kernel size

        #convolutional layer 2: intakes 14x14x14 tensor
                #inputs conv1 output
                #28 filters
                #same kernel size
        self.conv2 = nn.Conv2d(14, 28, 3, padding=2) 
        #filters = size after conv = (14-3+(2*2))/1 + 1 = 16, output size: 16*16*28

        #pooling layer: 16*16*28 -> 8*8*28

        # convolutional layer 3: intakes 8x8x28 tensor
                #inputs conv2 output
                #56 filters 
                #same kernel size
        self.conv3 = nn.Conv2d(28, 56, 3, padding=1)
        #filters = size after conv = (8-3+(2*1))/1 + 1 = 8, output size: 8*8*56


        # max pooling layer
        # 2x2 filter, shift across image every step
        self.pool = nn.MaxPool2d(2, 2)
            #output -> 3*3*56

        # linear layer (56 * 3 * 3 -> 500)
        self.fc1 = nn.Linear(56 * 4 * 4, 500)
        # linear layer (500 -> 10) 
        # filters into 10 layers for each class
        self.fc2 = nn.Linear(500, 10)

        # dropout layer (p=0.25) 
        #introduces noise into image and makes network more robust
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # relu each convolutional layer
        # then pool each layer
        x = self.pool(F.relu(self.conv1(x))) # output size: 28*28*14, pool=14*14*14
        x = self.pool(F.relu(self.conv2(x))) # output size: 16*16*28, pool=8*8*28
        x = self.pool(F.relu(self.conv3(x))) # output size: 8*8*56,   pool = 4*4*56

        # flatten image input
        x = x.view(-1, 4*4*56)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))# 64 * 4 * 4 -> 500
        # add dropout layer 
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x) # 500 -> 10 layers
        return x
    

#create CNN model
model = Net()

#move model to gpu if CUDA available
model.to(device)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)    



#training Loop
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):
    # keep variables to track loss
    train_loss = 0.0
    valid_loss = 0.0

    #Training loop
    model.train()
    for data, target in train_loader: 
        # Move tensors to device
        data = data.to(device)
        target = target.to(device)
        

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # print(f'data: {data.shape},  target: {target.shape}')

        # calculate the batch loss
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # update training loss
        train_loss += loss.item()*data.size(0)#batch size



    #model validation
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        data = data.to(device)
        target = target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)#batch size

    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    print(f'Average loss: {batch_size}')
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print("---------------------------------------------------------------")

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_FMNIST.pt') #save model
        valid_loss_min = valid_loss

#load model with lowest validation loss
model.load_state_dict(torch.load('model_FMNIST.pt'))

#Visualize test result
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)


# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 10))
for idx in np.arange(40):
    ax = fig.add_subplot(4, int(40/4), idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
    
        