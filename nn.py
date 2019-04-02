import numpy as np
import torch.nn as nn
import torch, torchvision

class NN(nn.Module):
    def __init__(self, n_state, n_actions):
        super(NN, self).__init__()
        self.hidden1 = nn.Linear(n_state, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, n_actions)
    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x
    def predict_proba(self, x):
        out = self.forward(x).detach().numpy()
        out -= np.min(out)
        return out / np.sum(out)
    def fit(self, x, y):
        optim = torch.optim.Adadelta(self.parameters(), lr = 0.007)
        loss_fn = nn.MSELoss()

        for s, a in zip(x, y):
            optim.zero_grad()
            out = self.forward(s)
            loss = loss_fn(out, torch.tensor(a, dtype = torch.float))
            loss.backward()

