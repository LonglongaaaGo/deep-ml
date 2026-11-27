import torch

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    if not vectors: return True

    mat = torch.as_tensor(vectors,dtype=torch.float64)
    m,n = mat.shape
    if m>n: return False

    eps = 1e-9
    rank = 0
    
    for col in range(n):
        if rank>=m: break
        abs_vals = mat[rank:,col].abs()
        idx = torch.argmax(abs_vals)
        pivot_idx = rank+idx

        if rank != pivot_idx:
            mat[[rank,pivot_idx]] = mat[[pivot_idx,rank]]

        high_val = mat[rank,col]
        if abs(high_val) <= eps: continue

        if rank+1<m:        
            factors = mat[rank+1:,col]/high_val
            mat[rank+1:,col:] = mat[rank+1:,col:] - factors.unsqueeze(1)*mat[rank,col:]

        rank+=1
    
    if rank==m: return True
    return False

        
        

            
    

 


        


