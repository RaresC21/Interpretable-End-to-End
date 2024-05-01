import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from interpretable import InterpretableForecast, SparseInterpretableForecast, AggregateInterpretableForecast

from problem import Problem
from data import DataGen
from models import * 

import matplotlib.pyplot as plt 

from helper import *

def eval_all(c): 
    two_stage_forecast, task_forecast, sparse_forecast, quantile_forecast = train(c)
    last_forecast = sparse_forecast

    print("TESTING -------------------------------------")
    two_cost, two_dec = eval_model(two_stage_forecast, X_test, Y_test, H, B, problem_.cross_costs, n_samples = 100)
    task_cost, task_dec = eval_model(task_forecast, X_test, Y_test, H, B, problem_.cross_costs, n_samples = 0)
    # sparse_cost, interpretable_dec = eval_model(sparse_forecast.predict, X_test, Y_test, H, B, problem_.cross_costs, n_samples = 0)
    aggregate_cost, interpretable_dec = eval_model(quantile_forecast.predict, X_test, Y_test, H, B, problem_.cross_costs, n_samples = 0)
    # quantile_cost = -1
    return two_cost, task_cost, 0, aggregate_cost

def train(c): 
    # cross_costs = np.array([[0, c], [c, 0]])
    cross_costs = np.ones((n_nodes, n_nodes)) * b
    for i in range(n_nodes): 
        for j in range(n_nodes): 
            cross_costs[i,j] = (abs(i - j)**c) * b / n_nodes

    problem_.set_cross_costs(cross_costs)

    print("Train end-to-end -------------------------------------")
    task_forecast = train_task_loss(problem_, X_train, Y_train, X_test, Y_test, two_stage_forecast, EPOCHS=2000)

    # print("Train quantile + linear -------------------------------------")
    # sparse_forecast = SparseInterpretableForecast(last_forecast, two_stage_forecast(X_train), 0.1, X_train, Y_train, problem_)
    # sparse_forecast.train(l1_reg = 0, EPOCHS=10000)

    print("Train linear + quantile -------------------------------------")
    quantile_forecast = AggregateInterpretableForecast(two_stage_forecast, two_stage_forecast(X_train), X_train, Y_train, problem_)
    quantile_forecast.train(l1_reg = 0, EPOCHS=10000)

    # quantile_forecast = None

    print()
    print("AGGREGATE FORECAST MODEL")
    print("mean:", quantile_forecast.mean_forecast)
    print("quantiles:", nn.functional.sigmoid(quantile_forecast.quantiles))
    print("var:", quantile_forecast.var_forecast)
    # print("quantiles", nn.functional.sigmoid(sparse_forecast.quantiles).detach().numpy())
    # print("combination", sparse_forecast.forecast.detach().numpy())
    print()

    return two_stage_forecast, task_forecast, 0, quantile_forecast

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(0)
    torch.manual_seed(0)

    # make random distances
    n_nodes = 4
    n_features = 10
    
    DEVICE = "cpu"
    if torch.cuda.is_available(): 
        DEVICE = "cuda"

    print("DEVICE", DEVICE)
        
    h = 1
    b = 5
    H = torch.tensor([h for i in range(n_nodes)]).to(DEVICE)
    B = torch.tensor([b for i in range(n_nodes)]).to(DEVICE)  

    problem_ = Problem(H, B, n_nodes)
    cross_costs = np.array([[0, 0], [0, 0]])
    
    n_data = 100
    n_test = 1000
    data_generator = DataGen(n_features, n_nodes, DEVICE)
    X_train, Y_train, X_test, Y_test = data_generator.get_test_train(n_data, n_test)
    
    print("Train Two-Stage -------------------------------------")
    two_stage_forecast = train_two_stage(problem_, X_train, Y_train, EPOCHS=1000, DEVICE=DEVICE)
    last_forecast = two_stage_forecast

    two_stage_costs = [] 
    task_costs = [] 
    sparse_costs = [] 
    aggregate_costs = []

    step = 1
    cost_range = np.arange(h,b+step,step)
    for c in cost_range:
        print("cross cost:", c)
        two_cost, task_cost, sparse_cost, aggregate_cost = eval_all(c)
        print("two-stage", np.mean(two_cost))
        print("task-cost", np.mean(task_cost))
        # print("sparse   ", np.mean(sparse_cost))
        print("quantile ", np.mean(aggregate_cost))
        print()

        two_stage_costs.append(np.mean(two_cost))
        task_costs.append(np.mean(task_cost))
        # sparse_costs.append(np.mean(sparse_cost))
        aggregate_costs.append(np.mean(aggregate_cost))

    plt.plot(cost_range, two_stage_costs, label='two-stage')
    plt.plot(cost_range, task_costs, label='task')
    # plt.plot(cost_range, sparse_costs, label='sparse')
    plt.plot(cost_range, aggregate_costs, label='aggregate')
    plt.legend()
    plt.show()
