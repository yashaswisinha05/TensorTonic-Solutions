import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method=="mse":
        return float(torch.mean((target-pred)**2))
    elif method=="cross_entropy":
        m=pred.max(dim=1,keepdim=True).values
        shifted=pred-m
        val=pred[torch.arange(pred.shape[0]), target]
        shifted_new=val-m.squeeze(1)
        # shifted_log=torch.exp()
        return float(torch.mean(torch.logsumexp(shifted,dim=1)-shifted_new))
        
    elif method=="huber":
        a=torch.abs(pred-target)
        return float(torch.mean(torch.where(a<=delta,1/2*(a**2),(delta*(a-0.5*delta)))))
        # if a<=delta:
        #     return 1/2*(a**2)
        # else:
        #     return (delta*(a-0.5*delta))
          
    pass
