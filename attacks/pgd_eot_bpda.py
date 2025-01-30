import numpy as np
import torch
import torch.nn.functional as F


class BPDA:
    def __init__(self, get_logit, attack_steps=200, eps=8./255., step_size=0.007, target=None, eot=20):
        self.target = target
        self.clamp = (0,1)
        self.eps = eps
        self.step_size = step_size
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        for _ in range(self.attack_steps):
            grad = torch.zeros_like(x_adv)
            
            for _ in range(self.eot):
                with torch.no_grad():
                    preprocessed_x = self.get_logit.preprocess(x_adv)
                preprocessed_x.requires_grad = True
                
                # Classification
                logits = self.get_logit.classify(preprocessed_x)
                
                # Calculate loss
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                grad += torch.autograd.grad(loss, [preprocessed_x])[0].detach()
                x_adv = x_adv.detach()

            grad /= self.eot
            grad = grad.sign()
            x_adv = x_adv + self.step_size * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv