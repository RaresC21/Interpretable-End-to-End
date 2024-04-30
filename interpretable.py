import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class InterpretableForecast(nn.Module):
    def __init__(self, base_forecast, mean, std, x_train, target, problem): 
        super(InterpretableForecast, self).__init__()

        self.base_forecast = base_forecast
        self.mean = mean
        self.std = std 
        self.problem = problem
        self.x_train = x_train
        self.target = target


        self.granularity = 0.01
        self.qs = np.expand_dims(np.arange(self.granularity,1, self.granularity), 1)
        self.quantiles = self.get_quantiles(mean, std)
        # q = self.get_quantiles(mean, std)
        # self.quantiles = q.reshape(len(q), q.shape[1] * q.shape[2])

        # self.weights = nn.parameter.Parameter(torch.rand(q.shape[1:]))
        self.forecast = nn.Linear(self.quantiles.shape[1], self.target.shape[1])
        # self.forecast = nn.Linear(self.mean.shape[1], self.target.shape[1])

    def get_quantiles(self, mean, std): 
        q = np.repeat(self.qs, len(mean), axis=1)
        q = np.expand_dims(q, 2)
        q = np.repeat(q, len(mean[0]), axis=2)
        quantiles = norm.ppf(q, loc = mean.detach().numpy(), scale = std)
        quantiles = np.swapaxes(quantiles, 0,1)
        # return quantiles
        return quantiles.reshape(len(quantiles), quantiles.shape[1] * quantiles.shape[2])

    def predict(self, x): 
        qs = torch.tensor(self.get_quantiles(self.base_forecast(x), 1)).float()
        # o = nn.functional.softmax(self.weights, dim=0) * qs
        # return self.forecast(o.reshape(o.shape[0], -1))
        return self.forecast(qs)

    def train(self, l1_reg = 10, EPOCHS = 200):

        best_error = 100000
        optimizer_task = optim.Adam(self.forecast.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        all_errs = []
        all_test_errs = []
        batch_size = 50
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            for i in range(0, self.target.size()[0], batch_size):
                d = torch.tensor(self.target[i:i+batch_size:]).float()
                input = torch.tensor(self.x_train[i:i+batch_size,:]).float()

                optimizer_task.zero_grad()


                f = self.predict(input)
                # f = self.forecast(input)

                all_linear1_params = torch.cat([x.view(-1) for x in self.forecast.parameters()])
                l1_regularization = l1_reg * torch.norm(all_linear1_params, 1)
                error1 = self.problem.fulfilment_loss(f, d) / len(d) 
                error = error1 + l1_regularization
                
                error.backward()
                optimizer_task.step()

                all_errs.append(error1.detach().numpy())
                
                # print(np.mean(all_errs))


from torch.distributions import Normal
from torch.autograd import Variable
class SparseInterpretableForecast(nn.Module):
    def __init__(self, base_forecast, mean, std, x_train, target, problem): 
        super(SparseInterpretableForecast, self).__init__()

        self.base_forecast = base_forecast
        self.mean = mean
        self.std = std 
        self.problem = problem
        self.x_train = x_train
        self.target = target

        self.granularity = 0.1
        self.qs = np.expand_dims(np.arange(self.granularity,1, self.granularity), 1)
        # q = self.get_quantiles(mean, std)
        # self.quantiles = q.reshape(len(q), q.shape[1] * q.shape[2])

        self.quantiles = Variable(torch.rand(target.shape[1]), requires_grad = True)
        # self.forecast = nn.Linear(self.quantiles.shape[0], self.target.shape[1])
        self.forecast = Variable(torch.rand(self.quantiles.shape[0], self.target.shape[1]), requires_grad = True)
        # self.weights = nn.parameter.Parameter(torch.rand(q.shape[1:]))
        # self.forecast = nn.Linear(self.quantiles.shape[1], self.target.shape[1])
        # self.forecast = nn.Linear(self.mean.shape[1], self.target.shape[1])

    def predict(self, x): 
        means = self.base_forecast(x)
        dist = Normal(means, self.std)
        qs = dist.icdf(nn.functional.sigmoid(self.quantiles))
        # qs = dist.icdf(self.quantiles)
        # qs = torch.tensor(self.get_quantiles(self.base_forecast(x), 1)).float()
        # o = nn.functional.softmax(self.weights, dim=0) * qs
        # return self.forecast(o.reshape(o.shape[0], -1))
        # print(qs.shape)
        # print(torch.sum(qs, dim=1).shape)
        # print(torch.sum(qs, dim=1).unsqueeze(1).shape)
        # print(torch.sum(qs, dim=1).unsqueeze(1).repeat(1,2).shape)
        # return torch.sum(qs, dim=1).unsqueeze(1).repeat(1,2)
        return F.relu((self.forecast @ qs.T).T)
        # return qs

    def train(self, l1_reg = 10, EPOCHS = 200):

        best_error = 100000
        # print(self.quantiles)
        # optimizer_task = optim.Adam([self.quantiles], lr=0.0001)
        optimizer_task = optim.Adam([self.forecast, self.quantiles], lr=0.0001)
        # optimizer_task.add_param_group({"params": self.quantiles})
        criterion = nn.MSELoss()

        all_errs = []
        all_test_errs = []
        batch_size = 50
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            for i in range(0, self.target.size()[0], batch_size):
                d = torch.tensor(self.target[i:i+batch_size:]).float()
                input = torch.tensor(self.x_train[i:i+batch_size,:]).float()

                optimizer_task.zero_grad()

                f = self.predict(input)
                # f = self.forecast(input)

                # all_linear1_params = torch.cat([x.view(-1) for x in self.forecast.parameters()])
                # l1_regularization = l1_reg * torch.norm(all_linear1_params, 1)
                error = self.problem.fulfilment_loss(f, d) / len(d) 
                # error = error1 + l1_regularization
                
                error.backward()
                # print("GRAD", self.quantiles.grad)

                optimizer_task.step()

                all_errs.append(error.detach().numpy())
                
                # print(np.mean(all_errs[-200:]))

            
from torch.distributions import Normal, MultivariateNormal
from torch.autograd import Variable
class AggregateInterpretableForecast(nn.Module):
    def __init__(self, base_forecast, mean, x_train, target, problem): 
        super(AggregateInterpretableForecast, self).__init__()

        self.base_forecast = base_forecast
        self.mean = mean
        self.problem = problem
        self.x_train = x_train
        self.target = target

        self.granularity = 0.1
        self.qs = np.expand_dims(np.arange(self.granularity,1, self.granularity), 1)
        # q = self.get_quantiles(mean, std)
        # self.quantiles = q.reshape(len(q), q.shape[1] * q.shape[2])

        self.quantiles = Variable(torch.rand(target.shape[1]), requires_grad = True)
        # self.mean_forecast = nn.Linear(self.quantiles.shape[0], self.target.shape[1])
        self.mean_forecast = Variable(torch.rand(self.target.shape[1], self.target.shape[1]), requires_grad = True)
        # self.mean_forecast.bias = torch.nn.Parameter(torch.zeros(1))
        # self.var_forecast = Variable(torch.ones(self.target.shape[1]) * 0.1, requires_grad = True)
        self.var_forecast = torch.ones(self.target.shape[1]) * 0.1
                        
        # self.weights = nn.parameter.Parameter(torch.rand(q.shape[1:]))
        # self.forecast = nn.Linear(self.quantiles.shape[1], self.target.shape[1])
        # self.forecast = nn.Linear(self.mean.shape[1], self.target.shape[1])

    def predict(self, x): 
        means = self.base_forecast(x)

        means = (self.mean_forecast @ means.T).T

        # print(mean.shape)
        # print(self.var_forecast.shape)
        v = self.var_forecast.unsqueeze(0).repeat(x.shape[0], 1)
        # print(v.shape)
        dist = Normal(means, v)
        qs = dist.icdf(nn.functional.sigmoid(self.quantiles))
        # print(self.mean_forecast.bias)
        # qs = dist.icdf(self.quantiles)
        # qs = torch.tensor(self.get_quantiles(self.base_forecast(x), 1)).float()
        # o = nn.functional.softmax(self.weights, dim=0) * qs
        # return self.forecast(o.reshape(o.shape[0], -1))
        # return self.forecast(qs)
        return qs

    def train(self, l1_reg = 10, EPOCHS = 200):

        best_error = 100000
        # print(self.quantiles)
        # optimizer_task = optim.Adam([self.quantiles], lr=0.0001)
        optimizer_task = optim.Adam([self.mean_forecast], lr=0.0001)
        optimizer_task.add_param_group({"params": self.quantiles})
        # optimizer_task.add_param_group({"params": self.var_forecast})
        criterion = nn.MSELoss()

        all_errs = []
        all_test_errs = []
        batch_size = 50
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            for i in range(0, self.target.size()[0], batch_size):
                d = torch.tensor(self.target[i:i+batch_size:]).float()
                input = torch.tensor(self.x_train[i:i+batch_size,:]).float()

                optimizer_task.zero_grad()

                f = self.predict(input)
                # f = self.forecast(input)

                # all_linear1_params = torch.cat([x.view(-1) for x in self.forecast.parameters()])
                # l1_regularization = l1_reg * torch.norm(all_linear1_params, 1)
                error = self.problem.fulfilment_loss(f, d) / len(d) 
                # error = error1 + l1_regularization
                
                error.backward()
                # print("GRAD", self.quantiles.grad)

                optimizer_task.step()


                all_errs.append(error.detach().numpy())
                
                # print(np.mean(all_errs[-200:]))

            