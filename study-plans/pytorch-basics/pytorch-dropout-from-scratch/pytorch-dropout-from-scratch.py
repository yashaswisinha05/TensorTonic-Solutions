import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p=p
        
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        multiplier = (torch.rand_like(x, dtype=torch.float32) > self.p).to(x.dtype)
        
        if self.p!=1 and self.training:
            y=(1/(1-self.p))*multiplier*x
        elif self.training!=True:
            y=x
        else:
            y=torch.zeros_like(x)
        return y
        pass
