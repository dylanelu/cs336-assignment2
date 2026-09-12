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


class FlattenedDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module: nn.Module = module

        with torch.no_grad():
            # Flatten all params send only one communication
            params = list(param for param in self.module.parameters())
            flat_params = torch._utils._flatten_dense_tensors(params)
            dist.broadcast(flat_params, src=0)

            # Set the parametres after unflattening
            unflattened_params = torch._utils._unflatten_dense_tensors(flat_params, params)
            for param, updated in zip(self.module.parameters(), unflattened_params):
                param.copy_(updated)

        

    def forward(self, data):
        return self.module(data)
    
    def finish_gradient_synchronization(self):
        # Flatten all grads send only one communication
        # Send only the non None grads and non frozen
        grads = list(param.grad for param in self.module.parameters() if param.requires_grad and param.grad is not None)
        if len(grads) == 0:
            return
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        
        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG, async_op=False)

        # Set the grads after unflattening
        unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, updated in zip(grads, unflattened_grads):
            grad.copy_(updated)


class OverlappedDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module: nn.Module = module
        self.hooks = []

        with torch.no_grad():
            # Flatten all params send only one communication
            params = list(param for param in self.module.parameters())
            flat_params = torch._utils._flatten_dense_tensors(params)
            dist.broadcast(flat_params, src=0)

            # Set the parametres after unflattening
            unflattened_params = torch._utils._unflatten_dense_tensors(flat_params, params)
            for param, updated in zip(self.module.parameters(), unflattened_params):
                param.copy_(updated)

            # Register the post grad hooks
            for param in params:
                if not param.requires_grad:
                    continue
                param.register_post_accumulate_grad_hook(lambda p: self.post_grad_hook(p))


    def post_grad_hook(self, param):
        # Launch the sync and store the hook
        h = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.hooks.append(h)

    def forward(self, data):
        return self.module(data)
    
    def finish_gradient_synchronization(self):
        # Wait for all handlers to finish
        # Set our local gradients
        for h in self.hooks:
            h.wait()
            
        self.hooks.clear()

