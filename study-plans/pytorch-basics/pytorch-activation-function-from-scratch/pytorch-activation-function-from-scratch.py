import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method =="relu":
        return torch.maximum(torch.zeros_like(x),x)
    elif method =="sigmoid":
        return 1/(1+torch.exp(-x))
    elif method =="tanh":
        
        return (torch.exp(x-torch.abs(x))-torch.exp(-x-torch.abs(x)))/(torch.exp(x-torch.abs(x))+torch.exp(-x-torch.abs(x)))
        #How to avoid explosion
        
    elif method =="leaky_relu":
        return torch.where(x>0,x,0.01*x)
    pass
