import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    if v1 == None or v2 == None:
        raise Error("cannot empty")
    if v1.ndim != v2.ndim:
        raise Error("should same shape")
    if v1.shape != v2.shape:
        raise Error("should same shape")

    v1 = v1.float()
    v2 = v2.float()
    # Implement your code here
    out = F.cosine_similarity(v1,v2,dim=0)
    return float(out.item())