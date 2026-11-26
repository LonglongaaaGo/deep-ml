import torch

def matrix_rank(A: torch.Tensor, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n) as a torch.Tensor
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """

    M = A.clone().to(torch.float64)
    m,n = M.shape
    rank = 0

    for col in range(n):
        if rank>= m: break
        sub_col = M[rank:,col].abs()
        max_idx = torch.argmax(sub_col).item()
        pivot_row = rank + max_idx
        if sub_col[max_idx] <=tol: continue

        if pivot_row != rank:
            M[[rank,pivot_row]] = M[[pivot_row,rank]]

        pivot_val = M[rank,col]
        if rank+1<m:
            factors = M[rank+1:,col]/pivot_val
            M[rank+1:,col:] =  M[rank+1:,col:] - factors.unsqueeze(1) * M[rank,col:]
        rank+=1

    return rank
    

    