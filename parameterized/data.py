import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DataGen:
    def __init__(self, n_features = 10, n_nodes = 10):
        self.n_features = n_features
        self.data_model = nn.Sequential(
            nn.Linear(n_features, n_nodes),
            nn.ReLU(),
        )

    def get_test_train(self, n_data, n_test = 1000): 
        Y_train, X_train = self.make_data(n_data)
        Y_test, X_test = self.make_data(n_test)    
        return X_train, Y_train, X_test, Y_test

    def make_data(self, n_data):
        # X2 = 2 * torch.randn(n_data, self.n_features) + 0.1
        # X1 = torch.randn(n_data, self.n_features) - 0.2
        # mu = torch.rand(n_data)
        # mu = mu.view(-1,1).repeat(1,self.n_features)
        # X = (X1 + X2 * mu)

        X = torch.randn(n_data, self.n_features)        
        y = self.data_model(X)

        # y += 0.1 * torch.from_numpy(np.random.multivariate_normal([0,0], [[1,0],[0,1]], len(y)))
        y += 0.1 * torch.randn_like(y)

        # andn(y.shape)
        # y = 0.1 + y.repeat(1, 2) + 0.0 * torch.randn(y.shape).repeat(1,2) * torch.tensor([1, -1])
        return F.relu(y), X
