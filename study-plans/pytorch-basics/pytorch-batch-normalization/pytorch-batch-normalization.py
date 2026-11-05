import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean_j=torch.mean(X,dim=0)
    sigma_j=torch.mean((X-mean_j)**2,dim=0)
    Y=gamma*((X-mean_j)/(torch.sqrt(sigma_j+eps)))+beta
    return Y
    pass
    
