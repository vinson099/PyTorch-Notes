import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad = torch.tensor([[0.5,2.0,0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)