from pandas.io.parsers import read_csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, art3d

df = read_csv('C:\\Users\\simon\\OneDrive\\Dokumenter\\NTNU\\Dataingeniør 2021-2022\\Høst\\IDATT2502 - Anvendt maskinlæring med prosjekt\\ObligsMachineLearning\\Oblig1\\day_length_weight.csv', skiprows=0)

# Reading data from csv file
x_l = np.array(df.iloc[:, 1].values)
y_w = np.array(df.iloc[:, 2].values)
z_day = df.iloc[:,0].values

# Reshaping data accordingly
lw = pd.DataFrame({'l':x_l, 'w':y_w })
x_train = torch.tensor(lw.to_numpy(), dtype=torch.float32).reshape(-1, 2)
y_train = torch.tensor([z_day], dtype=torch.float32).reshape(-1 , 1)

class LinearRegressionModel:
    def __init__(self): 
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)

for epoch in range(1000000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

ax = plt.figure().add_subplot(111, projection='3d')
ax.set_xlabel("length")
ax.set_ylabel("weight")
ax.set_zlabel("days")  

ax.scatter(x_l, y_w, z_day, c='r', marker='o')

x = torch.arange(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 5)
y = torch.arange(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 5)

x_surf, y_surf = torch.meshgrid(x, y)

zs = model.f(torch.cat((torch.ravel(x_surf).unsqueeze(1), torch.ravel(y_surf).unsqueeze(1)), dim = -1).type(torch.FloatTensor))
zz = zs.reshape(x_surf.shape).detach()

ax.plot_wireframe(x_surf, y_surf, zz)
plt.show()
