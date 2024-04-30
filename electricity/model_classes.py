#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce

from torch.distributions import Normal, MultivariateNormal
from torch.autograd import Variable


import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim

from qpth.qp import QPFunction
from constants import *

def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0)).mean(0)

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
    def forward(self, x):
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)


class InterpretableNet(nn.Module): 
    def __init__(self, X, Y, base_forecast, params): 
        super(InterpretableNet, self).__init__()
        self.X = X 
        self.Y = Y 
        self.base_forecast = base_forecast
        self.params = params
        self.d = params['n']

        self.quantiles = Variable(torch.rand(self.Y.shape[1]), requires_grad = True)
        self.mean_forecast = Variable(torch.rand(self.Y.shape[1], self.Y.shape[1]), requires_grad = True)
        self.var_forecast = Variable(torch.ones(self.Y.shape[1]) * 0.1, requires_grad = True)
    
        self.solver = SolvePointQP(params)

    def predict(self, x): 
        means = self.base_forecast(x)[0]
        means = means + (self.mean_forecast @ means.T).T / (self.d**2)

        v = self.var_forecast.unsqueeze(0).repeat(x.shape[0], 1)
        dist = Normal(means, v)
        qs = dist.icdf(nn.functional.sigmoid(self.quantiles))

        return qs

    def train(self, lr=1e-4, EPOCHS = 20):

        optimizer_task = optim.Adam([self.mean_forecast, self.quantiles], lr=lr)
        criterion = nn.MSELoss()

        all_errs = []
        all_test_errs = []
        batch_size = 50
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            for i in range(0, self.Y.shape[0], batch_size):
                d = torch.tensor(self.Y[i:i+batch_size:]).float()
                input = torch.tensor(self.X[i:i+batch_size,:]).float()

                optimizer_task.zero_grad()

                f = self.predict(input)
                
                decision = self.solver(f)[:,:self.params["n"]]

                # print("decision:", decision[0])
                # print("pred:", f[0])

                error = task_loss(decision, d, self.params) / len(d) 

                error.mean().backward()
                optimizer_task.step()
                all_errs.append(error.detach().numpy())
            if epoch % 1 == 0: 
                print("epoch", epoch, ":", np.mean(all_errs[-100:]))


def GLinearApprox(gamma_under, gamma_over):
    """ Linear (gradient) approximation of G function at z"""
    class GLinearApproxFn(Function):
        @staticmethod    
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.cdf(
                z.cpu().numpy()) - gamma_under)
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = (gamma_under + gamma_over) * pz
            dmu = -dz
            dsig = -(gamma_under + gamma_over)*(z-mu) / sig * pz
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GLinearApproxFn.apply


def GQuadraticApprox(gamma_under, gamma_over):
    """ Quadratic (gradient) approximation of G function at z"""
    class GQuadraticApproxFn(Function):
        @staticmethod
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.pdf(
                z.cpu().numpy()))
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = -(gamma_under + gamma_over) * (z-mu) / (sig**2) * pz
            dmu = -dz
            dsig = (gamma_under + gamma_over) * ((z-mu)**2 - sig**2) / \
                (sig**3) * pz
            
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GQuadraticApproxFn.apply

class SolvePointQP(nn.Module): 
    def __init__(self, params, DEVICE='cuda'):
        super(SolvePointQP, self).__init__()

        print("JIIII")

        self.DEVICE=DEVICE
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        self.n_vars = self.n * 3

        G = []
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = -1 
            cur[i + self.n] = -1
            G.append(cur)
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + self.n * 2] = -1 
            G.append(cur)
        for i in range(self.n-1): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + 1] = -1
            G.append(cur)

            cur = [0] * self.n_vars 
            cur[i] = -1
            cur[i + 1] = 1
            G.append(cur)

        for i in range(self.n_vars): 
            cur = [0] * self.n_vars 
            cur[i] = -1
            G.append(cur)

        self.G = torch.tensor(G, device=DEVICE).double() 
        self.e = torch.DoubleTensor(device=DEVICE)

        self.Q = torch.eye(self.n_vars, device=DEVICE).double() * 1e-3
        # for i in range(self.n): 
        #     self.Q[i,i] = 1e-3
        self.p = torch.tensor([0] * self.n + [params['gamma_under'] for _ in range(self.n)] + [params['gamma_over'] for _ in range(self.n)], device=DEVICE).double()
        # self.ramp_h =

    def forward(self, pred):
        nBatch, n = pred.shape

        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))

        ramp_h = torch.tensor(self.c_ramp * torch.ones((self.n - 1) * 2), device=DEVICE).double()
        ramp_h = self.ramp_h.unsqueeze(0).expand(nBatch, ramp_h.size(0))

        # print(pred.shape, ramp_h.shape, nBatch, self.n_vars)

        print(pred.get_device(), ramp_h.get_device())
        h = torch.cat((-pred, pred, ramp_h, torch.zeros(nBatch, self.n_vars, device=self.DEVICE)), 1)

        # print(h.shape, self.G.shape)

        return QPFunction(verbose=False)(self.Q, self.p, G, h, self.e, self.e)

class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        self.n_vars = self.n * 3

        G = []
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = -1 
            cur[i + self.n] = -1
            G.append(cur)
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + self.n * 2] = -1 
            G.append(cur)
        for i in range(self.n-1): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + 1] = -1
            G.append(cur)

            cur = [0] * self.n_vars 
            cur[i] = -1
            cur[i + 1] = 1
            G.append(cur)

        for i in range(self.n_vars): 
            cur = [0] * self.n_vars 
            cur[i] = -1
            

        self.G = torch.tensor(G).float() 
        self.e = torch.DoubleTensor()



    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) 
            for i in range(nBatch)], 0).double()
        p = (dg - d2g*z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out


class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """
    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.cuda()
        
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        
        # Find the solution via sequential quadratic programming, 
        # not preserving gradients
        z0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        mu0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        sig0 = sig.detach() # Variable(1. * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0-z0_new).norm().item()
            print("+ SQP Iter: {}, Solution diff = {}".format(i, solution_diff))
            z0 = z0_new
            if solution_diff < 1e-6:
                break
                  
        # Now that we found the solution, compute the gradient-propagating 
        # version at the solution
        dg = GLinearApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)
