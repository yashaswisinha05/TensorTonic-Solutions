import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.H1=nn.Linear(in_features,hidden_size)
        self.Y1=nn.Linear(hidden_size,out_features)
        self.act=nn.ReLU()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        
        H=self.act(self.H1(x))
        
        
        return self.Y1(H)
        pass
