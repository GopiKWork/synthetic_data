import torch
import torch.nn as nn
import pandas as pd
import numpy as np
class BayesianNetwork(nn.Module):
    def __init__(self, node_info):
        super(BayesianNetwork, self).__init__()
        self.node_info = node_info
        self.params = nn.ParameterDict()
        
        for node, parents in self.node_info.items():
            if not parents: #root nodes
                self.params[f"{node}_mean"] = nn.Parameter(torch.zeros(1)) #std gaussian distrib
                self.params[f"{node}_var"] = nn.Parameter(torch.ones(1)) #std gaussian distrib
            else: #child nodes
                num_parents = len(parents)
                self.params[f"{node}_beta"] = nn.Parameter(torch.zeros(num_parents + 1))
                self.params[f"{node}_res_var"] = nn.Parameter(torch.ones(1))
    
    def forward(self, data):
        samples = {}
        for node, parents in self.node_info.items():
            if not parents:
                mean = self.params[f"{node}_mean"].data
                std_dev = np.sqrt(self.params[f"{node}_var"].data)
                size_dim = (len(data[node]),)
                samples[node] = torch.normal(mean, std_dev,size=size_dim).numpy()
            else:
                X = torch.column_stack([data[parent] for parent in parents]) #all independent 
                X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1) #add intercept
                beta = self.params[f"{node}_beta"]
                res_var = self.params[f"{node}_res_var"]
                mean = X @ beta
                std_dev = torch.sqrt(res_var)
                samples[node] = torch.normal(mean, std_dev).numpy() #conditional distribution
        return samples
    
    def fit(self, data):
        for node, parents in self.node_info.items():
            node_data = torch.tensor(data[node].values, dtype=torch.float32)
            if not parents:
                # For root nodes, calculate mean and variance
                self.params[f"{node}_mean"].data = torch.mean(node_data)
                self.params[f"{node}_var"].data = torch.var(node_data)
            else:
                # For child nodes, calculate conditional distribution
                X = torch.column_stack(tuple( torch.tensor(data[parent].values, dtype=torch.float32) for parent in parents))
                X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
                y = node_data
                beta = torch.linalg.inv(X.T @ X) @ X.T @ y
                y_pred = X @ beta
                res_var = torch.mean((y - y_pred)**2)
                self.params[f"{node}_beta"].data = beta
                self.params[f"{node}_res_var"].data = res_var
    
    def sample(self, n_samples):
        with torch.no_grad():
            data = {}
            for node in self.node_info.keys():
                data[node] = torch.zeros(n_samples)
            samples = self.forward(data)
            samples_df = pd.DataFrame(samples)
        return samples_df
