
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from problem import Problem
from data import DataGen

from parameterized import ParameterizedExplainer

from models import * 

np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    n_nodes = 2
    n_features = 1

    h = 1
    b = 5
    H = torch.tensor([h for i in range(n_nodes)])
    B = torch.tensor([b for i in range(n_nodes)])  

    problem_ = Problem(H, B, n_nodes)

    n_data = 1000
    n_test = 1000
    data_generator = DataGen(n_features, n_nodes)
    X_train, Y_train, X_test, Y_test = data_generator.get_test_train(n_data, n_test)

    # base forecast
    print("training mse")
    two_stage_forecast = train_two_stage(problem_, X_train, Y_train, X_test, Y_test, EPOCHS=270)

    print() 
    print("training paraeterized model")
    model = ParameterizedExplainer(two_stage_forecast, X_train, Y_train, problem_)
    model.train(EPOCHS = 3000, lr=1e-4)

    print("base forecast param:", model.mean_forecast.detach().numpy())
    print("parameterization:   ", model.mean_forecast_base.detach().numpy())
    print("quantiles:          ", nn.functional.sigmoid(model.quantiles).detach().numpy())