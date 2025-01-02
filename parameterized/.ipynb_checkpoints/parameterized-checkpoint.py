import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
            
from torch.distributions import Normal, MultivariateNormal
from torch.autograd import Variable
import copy 

from constants import *


class ParameterizedExplainer(nn.Module):
    def __init__(self, base_forecast, x_train, target, problem): 
        super(ParameterizedExplainer, self).__init__()

        self.base_forecast = base_forecast
        self.problem = problem
        self.x_train = x_train
        self.target = target
        self.n_nodes = target.shape[1]

        self.quantiles = Variable(torch.rand(target.shape[1]), requires_grad = True)

        self.mean_forecast_base = Variable(torch.rand(self.target.shape[1], self.target.shape[1]), requires_grad = True)
        self.mean_forecast = Variable(torch.rand(self.target.shape[1], self.target.shape[1]), requires_grad = True)

        self.var_forecast = torch.ones(self.target.shape[1]) * 0.1
        self.all_ordered = problem.all_ordered
        self.all_cross_costs = problem.all_cross_costs
        self.c_values = problem.c_values
        
        self.hidden_dim = 1000
        
        n_features = x_train.shape[1]
        self.linear1 = nn.Linear(n_features, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.n_nodes)

        self.p_weight = Variable(torch.rand(self.n_nodes, self.hidden_dim, device=DEVICE), requires_grad = True)
        self.b_weight = Variable(torch.rand(self.n_nodes, self.hidden_dim, device=DEVICE), requires_grad = True)
            

#     def predict(self, x, c): 
#         means = self.base_forecast(x)

#         means = ((self.mean_forecast * c + self.mean_forecast_base) @ means.T).T

#         v = self.var_forecast.unsqueeze(0).repeat(x.shape[0], 1) 

#         dist = Normal(means, v)
#         qs = dist.icdf(nn.functional.sigmoid(self.quantiles))

#         return qs

    def predict(self, x, c): 
        c /= self.c_values.max()
        f = F.relu(self.linear1(x))
        f2 = F.relu(self.linear2(f))
        
        A = self.p_weight * c + self.b_weight       
        out = (A @ f.T).T / self.hidden_dim
        return F.relu(out) + f2


    def train(self, x_val, y_val, lr=1e-4, l1_reg = 10, EPOCHS = 10000):
        b = self.problem.B[0]
        
        best_error = 100000
        # optimizer_task = optim.Adam([self.mean_forecast, self.mean_forecast_base, self.quantiles], lr=0.0001)
        optimizer_task = optim.Adam([self.linear1.weight, self.linear1.bias, self.p_weight, self.b_weight], lr=lr)


        all_errs = []
        all_test_errs = []
        batch_size = 50
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            for i in range(0, self.target.size()[0], batch_size):
                # c = np.random.rand() * self.problem.B[0]
                c_indx = np.random.randint(0,len(self.c_values))
                cross_costs = self.all_cross_costs[c_indx]
                cross_costs_ordered = self.all_ordered[c_indx] 
                c = self.c_values[c_indx]

                d = self.target[i:i+batch_size:]
                inp = self.x_train[i:i+batch_size,:]

                optimizer_task.zero_grad()

                f = self.predict(inp, c)
                error = self.problem.fulfilment_loss(f, d, cross_costs, cross_costs_ordered) 
                error.backward()

                optimizer_task.step()

                all_errs.append(error.item())
            if epoch % 50 == 0:

                test_cost = self.problem.evaluate_param(self.predict, x_val, y_val)[1]
                print("epoch ", epoch, "test cost: ", test_cost, best_error)
                if best_error > test_cost: 
                    best_error = test_cost
                    best_model_state = copy.deepcopy(self.state_dict())
                # else: 
                #     self.load_state_dict(best_model_state)
                #     return
        self.load_state_dict(best_model_state)


