import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    if not matrix.is_floating_point() and not matrix.is_complex():
        matrix = matrix.float()
    
    det_ = torch.linalg.det(matrix)
    trace_ = torch.trace(matrix)

    return det_,trace_

