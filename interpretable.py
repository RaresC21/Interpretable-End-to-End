import numpy as np

from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
            
from torch.distributions import Normal, MultivariateNormal
from torch.autograd import Variable

from helper import get_loader

class AggregateInterpretableForecast(nn.Module):
    def __init__(self, base_forecast, mean, problem, DEVICE="cpu"): 
        super(AggregateInterpretableForecast, self).__init__()

        self.base_forecast = base_forecast
        self.mean = mean
        self.problem = problem
        
        self.quantiles = torch.tensor(torch.rand(self.mean.shape[1]), requires_grad = True, device=DEVICE)
        self.mean_forecast = torch.tensor(torch.rand(self.mean.shape[1], self.mean.shape[1]), requires_grad = True, device=DEVICE)
        self.var_forecast = torch.ones(self.mean.shape[1]).to(DEVICE) * 0.1
                        
    def forward(self, x): 
        means = self.base_forecast(x)
        means = (self.mean_forecast @ means.T).T / len(means[0])

        v = self.var_forecast.unsqueeze(0).repeat(x.shape[0], 1)
        dist = Normal(means, v)
        qs = dist.icdf(nn.functional.sigmoid(self.quantiles))

        return qs

    
def train_interpretable(model, X_train, Y_train, lr=1e-4, EPOCHS = 200):
    best_error = 100000

    training_loader = get_loader(X_train, Y_train)

    optimizer_task = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    all_errs = []
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for data in training_loader:
            inp, d = data

            optimizer_task.zero_grad()

            f = model(inp)

            error = model.problem.fulfilment_loss(f, d) / len(d) 

            error.backward()
            optimizer_task.step()

            all_errs.append(error.cpu().detach().numpy())

        # if epoch % 10 == 0: 
        #     print("epoch:", epoch, " ", np.mean(all_errs[-100:]))
