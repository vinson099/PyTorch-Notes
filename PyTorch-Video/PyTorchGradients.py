import torch
import torch.nn as nn
	
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape #x.shape will return (4,1)
    #samples will have 4
    #features will have 1

input_size = n_features
output_size = n_features

#pass in features into model
model = nn.Linear(input_size, output_size)
    #takes input size and output size   

#what if we need a custom size for model
class LinearRegression(nn.Module):
        
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__iinit()
            #define layers
            self.lin = nn.Linear(input_dim, output_dim)

        def forawrd(self, x):
             return self.lin(x)



print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
		#accuracy to 3 decimal places
        #use .item() so we see the actual value without the tensor descriptor



#Training
learning_rate = 0.01
n_iters = 10000 #number of iterations

loss = nn.MSELoss() #calculate loss from Mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)
    # print(y_pred)

    #loss
    l = loss(Y, y_pred)
    # print(l)

    #gradients = backward pass
    l.backward() #uses pytorch in order to auto calculate gradient

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if epoch%10 == 0: #every time epoch loops
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

#After training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
