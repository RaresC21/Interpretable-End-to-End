import torch 
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from constants import * 

class Problem: 
    def __init__(self, H, B, n_nodes): 
        self.H = H
        self.B = B 
        self.n_nodes = n_nodes

        b = B[0].item()
        self.all_cross_costs = []
        self.all_ordered = []
        self.c_values = np.arange(0,b,0.1)
        for i in range(len(self.c_values)):
            c = self.c_values[i]
            cross_costs = torch.ones((self.n_nodes, self.n_nodes), device=DEVICE) * b
            for i in range(self.n_nodes): 
                for j in range(self.n_nodes): 
                    cross_costs[i,j] = (abs(i - j)*c) * b / self.n_nodes
            cross_costs_ordered = self.get_cross_costs_ordered(cross_costs.cpu().numpy())
            self.all_cross_costs.append(cross_costs)
            self.all_ordered.append(cross_costs_ordered)
            
    def evaluate(self, model, X_test, Y_test):
        costs = []
        for c, o in zip(self.all_cross_costs, self.all_ordered):
            pred = model(X_test) 
            cost = self.fulfilment_loss(pred, Y_test, c, o).item()
            costs.append(cost)
        return costs, np.mean(costs)   
    
    def evaluate_param(self, model, X_test, Y_test): 
        costs = []
        for c, o in zip(self.all_cross_costs, self.all_ordered):
            cross_cost = c[1][0].item()
            pred = model(X_test, cross_cost) 
            cost = self.fulfilment_loss(pred, Y_test, c, o).item()
            costs.append(cost)
        return costs, np.mean(costs)   

    
    def get_cross_costs_ordered(self, costs):
        cross_costs_ordered = [(costs[i,j], (i,j)) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        cross_costs_ordered = sorted(
            cross_costs_ordered,
            key=lambda x: x[0]
        )
        return cross_costs_ordered

    def fulfilment_loss(self, q, d, cross_costs, cross_costs_ordered):
        aa = torch.zeros(q.shape[0], self.n_nodes, self.n_nodes, device=DEVICE)
        qq = torch.clone(q).to(DEVICE) * 0
        dd = torch.clone(d).to(DEVICE) * 0

        for c, (i,j) in cross_costs_ordered:
            aa[:, i,j] = torch.minimum(q[:, i] - qq[:, i], d[:, j] - dd[:, j])
            qq[:, i] += aa[:, i,j] 
            dd[:, j] += aa[:, i,j]

        holding_cost = torch.sum(torch.sum(self.H * (q - torch.sum(aa, dim=2)), dim = 1), dim = 0)
        backorder_cost = torch.sum(torch.sum(self.B * (d - torch.sum(aa, dim=1)), dim = 1), dim = 0)
        edge_cost = torch.sum(aa * cross_costs)
        return (holding_cost + backorder_cost + edge_cost) / q.shape[0]


    def exact_loss(self, q, d, cross_costs):
        with torch.no_grad():
            aa, val = self.single(q, d)

        holding_cost = torch.sum(torch.sum(self.H * (q - torch.sum(aa, dim=2)), dim = 1), dim = 0)
        backorder_cost = torch.sum(torch.sum(self.B * (d - torch.sum(aa, dim=1)), dim = 1), dim = 0)
        edge_cost = torch.sum(aa * cross_costs)
        return aa, holding_cost + backorder_cost + edge_cost

    def single(self, q, demands): 
        K = demands.shape[0]
        N = demands.shape[1] 

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as solver:
                a_ = [(k,i,j) for k in range(K) for i in range(N) for j in range(N)]
                a = solver.addVars(a_, lb = 0, name='a')

                for k in range(K): 
                    for i in range(N): 
                        solver.addConstr(sum(a[k,i,j] for j in range(N)) <= q[k,i])
                        solver.addConstr(sum(a[k,j,i] for j in range(N)) <= demands[k,i])
                solver.setObjective(
                        sum( 
                            self.H[i] * sum(q[k,i].item() - sum(a[k,i,j] for j in range(N)) for i in range(N)) +
                            self.B[i] * sum(demands[k,i].item() - sum(a[k,j,i] for j in range(N)) for i in range(N)) +
                            sum(self.cross_costs[i,j].item() * a[k,i,j] for i in range(N) for j in range(N))
                            for k in range(K)
                            ) 
                        , GRB.MINIMIZE)
                
                solver.optimize()
                aa = []
                for k in range(K): 
                    all_cur = []
                    for i in range(N): 
                        cur = []
                        for j in range(N):
                            cur.append(a[k,i,j].X) 
                        all_cur.append(cur) 
                    aa.append(all_cur)
                return torch.tensor(aa), solver.getObjective().getValue() / N
