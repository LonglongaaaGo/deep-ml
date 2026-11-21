import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a = torch.as_tensor(a,dtype = torch.float)
    b = torch.as_tensor(b,dtype = torch.float)

    if a.ndim !=2:
        raise ValueError(" no ")
    if b.ndim !=2:
        raise ValueError(" no ")

    if a.shape[1] != b.shape[0]:
        return torch.tensor(-1)
    

    c = a @ b
    return c