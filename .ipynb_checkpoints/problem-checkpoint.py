import torch 
import numpy as np

class Problem: 
    def __init__(self, H, B, n_nodes, DEVICE='cpu'): 
        self.H = H
        self.B = B 
        self.n_nodes = n_nodes
        self.DEVICE=DEVICE
        
        self.cross_costs = []
        self.set_random_cross_costs()

    def set_random_cross_costs(self):
        cross_costs = np.random.rand(self.n_nodes, self.n_nodes)
        cross_costs = (cross_costs + cross_costs.T) / 2
        for i in range(self.n_nodes): cross_costs[i,i] = 0

        self.set_cross_costs(cross_costs)

    def set_cross_costs(self, costs):
        cross_costs_ordered = [(costs[i,j], (i,j)) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        self.cross_costs_ordered = sorted(
            cross_costs_ordered,
            key=lambda x: x[0]
        )
        self.cross_costs = torch.tensor(costs).to(self.DEVICE)

    def fulfilment_loss(self, q, d):
        aa = torch.zeros(q.shape[0], self.n_nodes, self.n_nodes).to(self.DEVICE)
        qq = torch.zeros_like(q).to(self.DEVICE)
        dd = torch.zeros_like(d).to(self.DEVICE)

        for c, (i,j) in self.cross_costs_ordered:
            aa[:, i,j] = torch.minimum(q[:, i] - qq[:, i], d[:, j] - dd[:, j])
            qq[:, i] += aa[:, i,j] 
            dd[:, j] += aa[:, i,j]

        holding_cost = torch.sum(torch.sum(self.H * (q - torch.sum(aa, dim=2)), dim = 1), dim = 0)
        backorder_cost = torch.sum(torch.sum(self.B * (d - torch.sum(aa, dim=1)), dim = 1), dim = 0)
        edge_cost = torch.sum(aa * self.cross_costs)
        return holding_cost + backorder_cost + edge_cost 


    def exact_loss(self, q, d):
        with torch.no_grad():
            aa, val = self.single(q, d)

        holding_cost = torch.sum(torch.sum(self.H * (q - torch.sum(aa, dim=2)), dim = 1), dim = 0)
        backorder_cost = torch.sum(torch.sum(self.B * (d - torch.sum(aa, dim=1)), dim = 1), dim = 0)
        edge_cost = torch.sum(aa * self.cross_costs)
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
