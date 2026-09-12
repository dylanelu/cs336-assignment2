import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module: nn.Module = module

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

    def forward(self, data):
        return self.module(data)
    
    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Average gradients
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
