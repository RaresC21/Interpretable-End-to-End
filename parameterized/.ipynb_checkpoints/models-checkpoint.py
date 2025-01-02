import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from constants import *
import copy

def eval_forecast_model(poblem_, X_train, model, D):    
    q = model(X_train) 
    return poblem_.fulfilment_loss(q, D).item()

class Forecast(nn.Module):
    def __init__(self, n_features, n_demands):
        super(Forecast, self).__init__()
    
        self.forecast = nn.Linear(n_features, n_demands)

    def forward(self, x): 
        return F.relu(self.forecast(x), inplace=False)


class Forecast2(nn.Module):
    def __init__(self, n_features, n_demands):
        super(Forecast2, self).__init__()
    
        self.linear1 = nn.Linear(n_features, 1000)
        self.linear2 = nn.Linear(1000, n_demands)

    def forward(self, x): 
        f = F.relu(self.linear1(x))
        f = F.relu(self.linear2(f))
        return f



def train_two_stage(problem_, X_train, Y_train, X_test, Y_test, EPOCHS = 2710):    
    best_error = 100000
    n_features = X_train.shape[1]
    n_nodes = Y_train.shape[1]

    two_stage_forecast = Forecast2(n_features, n_nodes)
    two_stage_forecast=two_stage_forecast.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer_twostage = optim.Adam(two_stage_forecast.parameters(), lr=0.0001)

    all_errs = []
    mses = []
    batch_size = 10
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for i in range(0, Y_train.size()[0], batch_size):
            c = Y_train[i:i+batch_size:]
            inp = X_train[i:i+batch_size,:]

            optimizer_twostage.zero_grad()

            f = two_stage_forecast(inp)

            mse = criterion(f, c) 

            mse.backward()
            optimizer_twostage.step()
            
            mses = np.append(mses, mse.item())

        if epoch % 50 == 0:
            # print(epoch, " ", np.mean(mses[-100:]))
            test_cost = problem_.evaluate(two_stage_forecast, X_test, Y_test)[1]
            print("epoch ", epoch, "test cost: ", test_cost)
            if best_error > test_cost + 1e-3: 
                best_error = test_cost
                best_model_state = copy.deepcopy(two_stage_forecast.state_dict())
            else: 
                two_stage_forecast.load_state_dict(best_model_state)
                two_stage_forecast.eval()
                return two_stage_forecast
            
    forecast.load_state_dict(best_model_state)
    return two_stage_forecast

def train_task_loss(problem_, X_train, Y_train, X_test, Y_test, cross_costs, cross_costs_ordered, EPOCHS = 200):   
    n_features = X_train.shape[1]
    n_nodes = Y_train.shape[1]
    forecast = Forecast2(n_features, n_nodes)
    forecast = forecast.to(DEVICE)
    
    best_error = 100000
    optimizer_task = optim.Adam(forecast.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    all_errs = []
    all_test_errs = []
    batch_size = 50
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for i in range(0, Y_train.size()[0], batch_size):
            d = Y_train[i:i+batch_size:]
            inp = X_train[i:i+batch_size,:]

            optimizer_task.zero_grad()

            f = forecast(inp)

            error = problem_.fulfilment_loss(f, d, cross_costs, cross_costs_ordered) 
            # mse = criterion(f, d)
            # error += mse * 10

            error.backward()
            optimizer_task.step()

            all_errs.append(error.item())
            
        if epoch % 50 == 0:
            test_cost = problem_.evaluate(forecast, X_test, Y_test)[1]
            print("epoch ", epoch, "test cost: ", test_cost)
            if best_error > test_cost: 
                best_error = test_cost
                best_model_state = copy.deepcopy(forecast.state_dict())
            # else: 
            #     forecast.load_state_dict(best_model_state)
            #     forecast.eval()
            #     return forecast
    forecast.load_state_dict(best_model_state)
    forecast.eval()
    return forecast



def train_exact_loss(poblem_, X_train, Y_train, X_test, Y_test, two_stage_forecast):
    n_features = X_train.shape[1]
    n_nodes = Y_train.shape[1]

    import copy 
    # forecast = copy.deepcopy(two_stage_forecast)
    forecast = Forecast(n_features, n_nodes)
    YY = Y_train.clone()

    best_error = 100000
    optimizer_task = optim.Adam(forecast.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    all_errs = []
    all_test_errs = []
    batch_size = 1
    for epoch in range(10):  # loop over the dataset multiple times
        for i in range(0, YY.size()[0], batch_size):
            d = torch.tensor(YY[i:i+batch_size:]).float()
            input = torch.tensor(X_train[i:i+batch_size,:]).float()

            optimizer_task.zero_grad()

            f = forecast(input)

            print('initial allocation', f)
            print('demand:', d)

            aa, error = poblem_.exact_loss(f, d) 
            # error = poblem_.exact_loss(f, d) / len(d)


            print("cost", error.item())

            print('fulfilment', aa)

            error.backward()
            optimizer_task.step()

            with torch.no_grad(): 

                f = forecast(input)

                print()
                print('new  allocation', f)

                aa, error = poblem_.exact_loss(f, d)
                print("new cost", error.item())
            print() 



            all_errs.append(error.detach().numpy())
        

        if epoch % 1 == 0:
            print("epoch:", epoch)
            print("Cost: ", np.mean(all_errs))
        #     test_errs = eval_forecast_model(poblem_, X_test, forecast, Y_test) / len(X_test)
        #     train_errs = eval_forecast_model(poblem_, X_train, forecast, Y_train) / len(X_train)
        #     print("epoch ", epoch, "test cost: ", test_errs.item(), "train cost: ", train_errs.item())
        #     all_test_errs.append(test_errs.item())

        #     if test_errs < best_error: 
        #         best_error = test_errs

    return forecast

def get_matrix_coos(m):
    dvars = m.getVars()
    constrs = m.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    for row_idx, constr in enumerate(constrs):
        for coeff, col_idx in get_expr_coos(m.getRow(constr), var_indices):
            yield row_idx, col_idx, coeff

def train_quad_loss(poblem_, X_train, Y_train, X_test, Y_test, two_stage_forecast): 
    import copy 
    forecast = copy.deepcopy(two_stage_forecast)
    YY = Y_train.clone()

    best_error = 100000
    optimizer_task = optim.Adam(forecast.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    all_errs = []
    all_test_errs = []
    batch_size = 50
    for epoch in range(200):  # loop over the dataset multiple times
        for i in range(0, YY.size()[0], batch_size):
            d = torch.tensor(YY[i:i+batch_size:]).float()
            input = torch.tensor(X_train[i:i+batch_size,:]).float()

            optimizer_task.zero_grad()

            f = forecast(input)

            error = poblem_.fulfilment_loss(f, d) / len(d)
            # mse = criterion(f, d)
            # error += mse * 10

            error.backward()
            optimizer_task.step()

            all_errs.append(error.detach().numpy())
            
        if epoch % 1 == 0:
            print("epoch:", epoch)
            print("Cost: ", np.mean(all_errs))
            test_errs = eval_forecast_model(poblem_, X_test, forecast, Y_test) / len(X_test)
            train_errs = eval_forecast_model(poblem_, X_train, forecast, Y_train) / len(X_train)
            print("epoch ", epoch, "test cost: ", test_errs.item(), "train cost: ", train_errs.item())
            all_test_errs.append(test_errs.item())

            if test_errs < best_error: 
                best_error = test_errs

    return forecast



# two-stage with distribution and saa

def saa(demands, holding, backorder, edges, q = None, test = False): 
    K = demands.shape[0]
    N = demands.shape[1] 

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as solver:
            a_ = [(k,i,j) for k in range(K) for i in range(N) for j in range(N)]
            q_ = [i for i in range(N)]
            a = solver.addVars(a_, lb = 0, name='a')
            if test is False: 
                q = solver.addVars(q_, lb = 0, name='q')
            m = solver.addVar(lb = 0)

            for k in range(K): 
                for i in range(N): 
                    solver.addConstr(sum(a[k,i,j] for j in range(N)) <= q[i])
                    solver.addConstr(sum(a[k,j,i] for j in range(N)) <= demands[k,i])
            solver.setObjective(
                    sum( # K
                        holding[0] * sum(q[i] - sum(a[k,i,j] for j in range(N)) for i in range(N)) +
                        backorder[0] * sum(demands[k,i] - sum(a[k,j,i] for j in range(N)) for i in range(N)) +
                        sum(edges[i][j] * a[k,i,j] for i in range(N) for j in range(N))
                        for k in range(K)) 
                    , GRB.MINIMIZE)
                
            solver.optimize()
            qq = []
            for v in solver.getVars():
                # print(v.VarName)
                if 'q' in v.VarName:
                    qq.append(v.X)
                # if 'a' in v.VarName: 
                #     print(v.VarName, v.X)
            return torch.tensor(qq).unsqueeze(0), solver.getObjective().getValue() / N