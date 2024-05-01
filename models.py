import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def eval_forecast_model(poblem_, X_train, model, D):    
    q = model(X_train) 
    return poblem_.fulfilment_loss(q, D) 

class Forecast(nn.Module):
    def __init__(self, n_features, n_demands, DEVICE="cpu"):
        super(Forecast, self).__init__()
    
        self.forecast = nn.Linear(n_features, n_demands, device=DEVICE)
        # self.linear = nn.Linear(n_features, 100)
        # self.forecast = nn.Linear(100, 2)

    def forward(self, x): 
        # f = F.relu(self.linear(x))
        # return F.relu(self.forecast(f)) 
        
        f = F.relu(self.forecast(x))
        return F.relu(f) + 0.05 
#         return newsvendor.get_cost_from_prob(f)

def train_two_stage(poblem_, X_train, Y_train, EPOCHS = 2710, DEVICE="cpu"):

    training_set = [[X_train[i], Y_train[i]] for i in range(len(X_train))]
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True)
    
    n_features = X_train.shape[1]
    n_nodes = Y_train.shape[1]

    YY = Y_train.clone().to(DEVICE)

    two_stage_forecast = Forecast(n_features, n_nodes).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer_twostage = optim.Adam(two_stage_forecast.parameters(), lr=0.001)

    all_errs = []
    mses = []
    batch_size = 10
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for data in training_loader:
            inp, c = data
 
            optimizer_twostage.zero_grad()

            f = two_stage_forecast(inp)

            mse = criterion(f, c) 
            mses.append(mse.cpu().detach().numpy())

            mse.backward()
            optimizer_twostage.step()
            
    return two_stage_forecast

def train_task_loss(poblem_, X_train, Y_train, X_test, Y_test, two_stage_forecast, EPOCHS = 200, DEVICE="cpu"):
    import copy 
    forecast = copy.deepcopy(two_stage_forecast)
    YY = Y_train.clone().to(DEVICE)

    best_error = 100000
    optimizer_task = optim.Adam(forecast.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    all_errs = []
    all_test_errs = []
    batch_size = 50
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for i in range(0, YY.size()[0], batch_size):
            d = YY[i:i+batch_size:]
            input = X_train[i:i+batch_size,:]

            optimizer_task.zero_grad()

            f = forecast(input)

            error = poblem_.fulfilment_loss(f, d) / len(d)
            # mse = criterion(f, d)
            # error += mse * 10

            error.backward()
            optimizer_task.step()

            all_errs.append(error.detach().numpy())

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