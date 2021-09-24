import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

class NOTOperator:
    def __init__(self): 
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)
   
    # Predictor
    def f(self, x):
        return (torch.sigmoid(x @ self.W + self.b))  

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)

model = NOTOperator()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 1)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()  
    optimizer.step()  
    optimizer.zero_grad() 

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))


ax = plt.figure().add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")  


x = torch.arange(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 0.1)
y = torch.arange(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 0.1)

x_surf, y_surf = torch.meshgrid(x, y)

zs = model.f(torch.cat((torch.ravel(x_surf).unsqueeze(1), torch.ravel(y_surf).unsqueeze(1)), dim = -1).type(torch.FloatTensor))
zz = zs.reshape(x_surf.shape).detach()

ax.scatter(x_train[:,0], x_train[:, 1], y_train, c='r', marker='o')

ax.plot_wireframe(x_surf, y_surf, zz)
plt.show()


