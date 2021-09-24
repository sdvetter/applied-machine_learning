import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd
import random

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class NOTOperator:
    def __init__(self): #2x2 
        self.W = torch.tensor([[random.uniform(0, 1), random.uniform(0, 1)],
         [random.uniform(0, 1), random.uniform(0, 1)]], requires_grad=True)  
        self.b = torch.tensor([[random.uniform(0, 1),random.uniform(0, 1)]], requires_grad=True)

        self.W2 = torch.tensor([[random.uniform(0, 1)] ,[random.uniform(0, 1)]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b2 = torch.tensor([[random.uniform(0, 1)]], requires_grad=True)
   
    # Predictor
    def f(self, x):
        return (torch.sigmoid(x @ self.W + self.b))  

    def f2(self, h):
        return torch.sigmoid(self.f(h) @ self.W2 + self.b2)

    
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f2(x), y)
        #-torch.mean(torch.multiply(y, torch.log(self.f2(x))) + torch.multiply((1 - y), torch.log(1 - self.f2(x))))

        #torch.nn.functional.binary_cross_entropy(self.pred(x), y)

model = NOTOperator()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b, model.W2, model.b2], 0000.1)

for epoch in range(10000):
    model.loss(x_train, y_train).backward()  
    optimizer.step()  
    optimizer.zero_grad() 

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))



x1 = x_train[:,0].view(-1,1).numpy()
x2 = x_train[:,1].view(-1,1).numpy()

x_surf, y_surf = torch.meshgrid(torch.arange(x1.min(), x1.max(), 0.1), torch.arange(x2.min(), x2.max(), 0.1))

oof = model.f2(torch.cat((torch.ravel(x_surf).unsqueeze(1), torch.ravel(y_surf).unsqueeze(1)), dim=-1).type(torch.FloatTensor))
z_surf = oof.reshape(x_surf.shape).detach()
'''
x = torch.arange(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 0.1)
y = torch.arange(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 0.1)
x_surf, y_surf = torch.meshgrid(x, y)
zs = model.f(torch.cat((torch.ravel(x_surf).unsqueeze(1), torch.ravel(y_surf).unsqueeze(1)), dim = -1).type(torch.FloatTensor))
zz = zs.reshape(x_surf.shape).detach()
'''

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_train[:,0], x_train[:, 1], y_train, c='r', marker='o')
ax.plot_wireframe(x_surf, y_surf, z_surf)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")  
#ax.plot_wireframe(x_surf, y_surf, zz)
plt.show()
