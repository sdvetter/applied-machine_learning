import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd

x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])

class NOTOperator:
    def __init__(self): 
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)
   
    # Predictor
    def f(self, x):
        return (torch.sigmoid(x @ self.W + self.b))  

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)

model = NOTOperator()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()  
    optimizer.step()  
    optimizer.zero_grad() 

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')

x = torch.arange(torch.min(x_train), torch.max(x_train), 0.01).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = sigmoid(xW + b)')

plt.legend()
plt.show()

