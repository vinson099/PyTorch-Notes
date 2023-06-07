import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#steps to follow


# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(
    n_samples = 100, n_features=1, noise=20, random_state=1)

#convert to tensor from numpy so we can use pytorch optimization and loss
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(X_numpy.astype(np.float32))

#want to reshape y so we can multiply X*y
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
    #X.shape will return (100,1)



# 1) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


# 2) Loss and optimizer
learning_rate = 0.01
#loss
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3) Training loop
    #forward pass
    #backwards pass
    #update weights

num_epochs = 100
for epoch in range(num_epochs):
    #forward pass & loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    #backwards pass
    loss.backward()

    #update
    optimizer.step()


    if(epoch + 1) % 10 ==0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f},')

    
#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, "b")
plt.show

#not sure why blueline is zero slope
