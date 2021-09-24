from pandas.io.parsers import read_csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd


df = read_csv('C:\\Users\\simon\\OneDrive\\Dokumenter\\NTNU\\Dataingeniør 2021-2022\\Høst\\IDATT2502 - Anvendt maskinlæring med prosjekt\\ObligsMachineLearning\\Oblig1\\length_weight.csv', skiprows=0)

x_axis = df.iloc[:,0].values
y_axis = df.iloc[:,1].values


x_train = torch.tensor([x_axis], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor([y_axis], dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self): 
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)

for epoch in range(1000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()

