import numpy as np

#create function that is a combination of weights and inputs
	# f = w * x
	# f = 2 * x    - > weights is 2
	
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#calculate model prediction
def forward(x):
	return w * x

#calculate loss
def loss(y, y_pred):
	return ((y_pred - y)**X).mean()


#calculate gradient
#MSE = 1/N * (w+x - y)**2
	#Mean Squared Error = 1/(Number of values) * 
	#							          (predicted value - actual)^2

#dJ/dw = 1/N 2x (w*x - y)
	#derivative of objective function with respect to weights = 
	#                        1/(Number of values) *

def gradient(x,y,y_pred):
	return np.dot(2*x, y_pred - y).mean()
	#dot product of yfunc and error


print(f'Prediction before training: f(5) = {forward(5):.3f}')
		#accuracy to 3 decimal places

#Training
learning_rate = 0.01
n_iters = 15 #number of iterations

for epoch in range(n_iters):
	#prediction = forward pass
	y_pred = forward(X)

	#loss
	l = loss(Y, y_pred)

	#gradients
	dw = gradient(X,Y,y_pred)

	#update weights
	w -= learning_rate * dw
		#learning rate * gradient

	if epoch%2 == 0: #every time epoch loops
		print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

#After training
print(f'Prediction after training: f(5) = {forward(5):.3f}')
