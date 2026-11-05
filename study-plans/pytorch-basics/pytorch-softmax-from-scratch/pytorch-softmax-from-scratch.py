import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max=torch.max(logits,dim=1,keepdim=True).values
    denominator=torch.sum(torch.exp(logits-max),dim=1,keepdim=True)
    
    p_ij=torch.exp(logits-max)/denominator
    return p_ij
    pass
