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

        b = self.problem.B[0]
        self.all_cross_costs = []
        self.all_ordered = []
        self.c_values = np.arange(0,b,0.1)
        for i in range(len(self.c_values)):
            c = self.c_values[i]
            cross_costs = torch.ones((self.n_nodes, self.n_nodes)) * b
            for i in range(self.n_nodes): 
                for j in range(self.n_nodes): 
                    cross_costs[i,j] = (abs(i - j)*c) * b / self.n_nodes
            cross_costs_ordered = self.problem.get_cross_costs_ordered(cross_costs.numpy())
            self.all_cross_costs.append(cross_costs)
            self.all_ordered.append(cross_costs_ordered)


    def predict(self, x, c): 
        means = self.base_forecast(x)

        means = ((self.mean_forecast * c + self.mean_forecast_base) @ means.T).T

        v = self.var_forecast.unsqueeze(0).repeat(x.shape[0], 1) 

        dist = Normal(means, v)
        qs = dist.icdf(nn.functional.sigmoid(self.quantiles))

        return qs

    def train(self, lr=1e-4, l1_reg = 10, EPOCHS = 10000):
        b = self.problem.B[0]
        
        best_error = 100000
        optimizer_task = optim.SGD([self.mean_forecast, self.mean_forecast_base, self.quantiles], lr=0.0001)

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

                d = torch.tensor(self.target[i:i+batch_size:]).float()
                input = torch.tensor(self.x_train[i:i+batch_size,:]).float()

                optimizer_task.zero_grad()

                f = self.predict(input, c)
                error = self.problem.fulfilment_loss(f, d, cross_costs, cross_costs_ordered) / (len(d)  * (1 + c))

                error.backward()

                optimizer_task.step()

                all_errs.append(error.detach().numpy())
            if epoch % 10 == 0:
                print("epoch", epoch, "err:", np.mean(all_errs[-100:]))

