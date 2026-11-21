import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    if a_t.ndim != 2:
        raise ValueError("should be dim = 2")
    a_t = torch.transpose(a_t,0,1)
    return a_t