import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    # Your implementation here
    if a.ndim != 2:
        raise ValueError("dim = 2")
    if b.ndim != 1:
        raise ValueError("dim = 1")
    if a.shape[1]!= b.shape[0]:
        return torch.tensor(-1)
    
    c = a @ b
    return c

