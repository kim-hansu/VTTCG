import math
import torch
from torch.optim.optimizer import Optimizer
import __main__
import numpy as np

class VTTCG(Optimizer):

    def __init__(self, params, lr=1e-3, beta=0.999, eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super(VTTCG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('VTTCG does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mt'] = group['eps'] * torch.zeros_like(p)
                    state['vt'] = torch.zeros_like(p)
                    state['og'] = group['eps'] * torch.zeros_like(grad)
                    state['max_vt'] = torch.zeros_like(p)

                mt, vt, og , max_vt = state['mt'], state['vt'], state['og'], state['max_vt']
                # Update the steps for each param group update
                state['step'] += 1
                beta = group['beta']


                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha = group['weight_decay'])

                pastck = (torch.sqrt(torch.sum(torch.pow(state['og'], 2))) * torch.sqrt(torch.sum(torch.pow(state['og'], 2))))
                if pastck < 1e-16:
                    pastck = 1e-16

                yk = grad.sub(og)
                betaprp = grad * yk / pastck
                theta = grad * mt / pastck

                mt = -grad + betaprp * mt - theta * yk
                vt = beta * vt.add_((1-beta) * torch.square(mt), alpha=1)
                torch.maximum(max_vt, vt, out=max_vt)

                lr_t = group['lr'] * (math.sqrt(1. - math.pow(beta, state['step'])))
                posts = 1 / (torch.sqrt(max_vt) + group['eps'])
                step_size = posts * lr_t

                p.add_(mt * step_size, alpha=1)

                og.copy_(grad)

            return

        return loss