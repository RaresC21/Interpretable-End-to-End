import numpy as np

import matplotlib.pyplot as plt

from problem import Problem
from data import DataGen

from parameterized import ParameterizedExplainer

from models import * 
from constants import * 


import torch

np.random.seed(0)
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")


def eval_models():     
    param_cost = []
    two_cost = [] 
    e2e_low = []
    e2e_high = [] 
    e2e_mid = [] 
    cross_costs_mult = []

    for c, o in zip(model.all_cross_costs, model.all_ordered):
        cross_cost = c[1][0].item()
        cross_costs_mult.append(cross_cost) 

        pred = model.predict(X_test, cross_cost)
        cost = problem_.fulfilment_loss(pred, Y_test, c, o).item()
        param_cost.append(cost)

        ts = two_stage_forecast(X_test)
        cost_two_stage = problem_.fulfilment_loss(ts, Y_test, c, o).item()
        two_cost.append(cost_two_stage)

        # e2 = end_to_end_forecast_low(X_test)
        # e2e_cost_low = problem_.fulfilment_loss(e2, Y_test, c, o).item()
        # e2e_low.append(e2e_cost_low)

        e2 = end_to_end_forecast_high(X_test)
        e2e_cost_high = problem_.fulfilment_loss(e2, Y_test, c, o).item()
        e2e_high.append(e2e_cost_high)

        # e2 = end_to_end_forecast_mid(X_test)
        # e2e_cost_mid = problem_.fulfilment_loss(e2, Y_test, c, o).item()
        # e2e_mid.append(e2e_cost_mid)
        
    return cross_costs_mult, param_cost, two_cost, e2e_high
        

if __name__ == '__main__':
    n_nodes = 5
    n_features = 5

    h = 1
    b = 10
    H = torch.tensor([h for i in range(n_nodes)], device=DEVICE)
    B = torch.tensor([b for i in range(n_nodes)], device=DEVICE)  

    problem_ = Problem(H, B, n_nodes)

    n_data = 1000
    n_test = 1000
    data_generator = DataGen(n_features, n_nodes)
    X_train, Y_train, X_test, Y_test = data_generator.get_test_train(n_data, n_test)
        

    EPOCHS = 3000
        
    print("training mse")
    two_stage_forecast = train_two_stage(problem_, X_train, Y_train, X_test, Y_test, EPOCHS=EPOCHS)
    
    print("task-loss")
    c1 = problem_.all_cross_costs[-1]
    c2 = problem_.all_ordered[-1]
    end_to_end_forecast_high = train_task_loss(problem_, X_train, Y_train, X_test, Y_test, c1, c2, EPOCHS=EPOCHS)

    print("training parameterized model")
    model = ParameterizedExplainer(data_generator.data_model, X_train, Y_train, problem_).to(DEVICE)
    model.train(X_test, Y_test, EPOCHS = EPOCHS, lr=1e-4)
        
    cross_costs_mult, param_cost, two_cost, e2e_high = eval_models()
    np.save("results/parameterized_model_costs.npy", param_cost)
    np.save("results/e2e_high_model_costs.npy", e2e_high)
    np.save("results/two_stage.npy", two_cost)
    
    plt.plot(cross_costs_mult, param_cost, label = 'param')
    plt.plot(cross_costs_mult, two_cost, label = 'two-stage')
    # plt.plot(cross_costs_mult, e2e_low, label = 'e2e-low')
    # plt.plot(cross_costs_mult, e2e_mid, label = 'e2e-mid')
    plt.plot(cross_costs_mult, e2e_high, label = 'e2e-high')
    plt.legend()
    plt.savefig('results/plot.png')