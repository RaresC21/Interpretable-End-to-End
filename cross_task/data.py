import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from constants import *

class DataGen:
    def __init__(self, n_features = 10, n_nodes = 10):
        self.n_features = n_features
        self.data_model = nn.Sequential(
            nn.Linear(n_features, n_nodes),
            nn.ReLU(),
        ).to(DEVICE)

    def get_test_train(self, n_data, n_test = 1000): 
        Y_train, X_train = self.make_data(n_data)
        Y_test, X_test = self.make_data(n_test)    
        return X_train, Y_train, X_test, Y_test

    def make_data(self, n_data):
        X = torch.randn(n_data, self.n_features).to(DEVICE)
        y = self.data_model(X).detach()

        y += 0.1 * torch.randn_like(y).to(DEVICE)

        return F.relu(y), X
