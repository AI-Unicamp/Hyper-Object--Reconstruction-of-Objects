import torch

class Example(torch.nn.Module):
    def __init__(self, param1=..., param2=...):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        out = ...
        return out
