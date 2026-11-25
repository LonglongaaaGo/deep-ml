import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    
    a,b,c,d = m[0,0], m[0,1], m[1,0], m[1,1]

    det_ = a*d - b*c

    if  det_ == 0:
        return None
    # m[0,0] = d/det_
    # m[0,1] = -b/det_
    # m[1,0] = -c/det_
    # m[1,1] = a/det_

    return torch.tensor([d,-b,-c,a],dtype=m.dtype,device = m.device)/det_

    








