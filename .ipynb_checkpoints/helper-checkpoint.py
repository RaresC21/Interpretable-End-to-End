import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from problem import Problem
from data import DataGen

def get_loader(X, y): 
    training_set = [[X[i], y[i]] for i in range(len(X))]
    return torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True)

def generate_samples(n_samples, means, rho, DEVICE="cpu"): 
    noises = torch.randn(n_samples, means.shape[0]).to(DEVICE) * rho
    return (torch.tensor(means) * ( 1 + noises)).detach().numpy()

def eval_model(forecast, X_test, Y_test, problem_):
    pred = forecast(X_test)
    return problem_.fulfilment_loss(pred, Y_test).cpu().detach().numpy() / len(pred)
    
#     all_costs = [] 
#     all_decisions = []

#     saa_rho = 0.1
#     saa_n_samples = n_samples

#     if n_samples == 0: 
#         saa_rho = 0 
#         saa_n_samples = 1

#     t = 0
#     for x,d in zip(X_test, Y_test): 
#         if t > 300: break
#         t += 1

#         pred = forecast(x.unsqueeze(0))
#         samples = generate_samples(saa_n_samples, pred, saa_rho, DEVICE=DEVICE)
#         decision, _ = saa(samples, H, B, cross_costs, test=False) 
#         # print(decision, samples)
#         _, cost = saa(d.unsqueeze(0).detach().numpy(), H, B, cross_costs, decision[0].detach().numpy(), test=True)
#         all_costs.append(cost)
#         # print(t, " ", np.mean(all_costs))
#         all_decisions.append(decision[0].numpy())
#     all_decisions = np.array(all_decisions)
#     all_decisions = torch.tensor(all_decisions) 
#     return all_costs, all_decisions