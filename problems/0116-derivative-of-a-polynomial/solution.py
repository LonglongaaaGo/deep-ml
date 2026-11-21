import torch

def poly_term_derivative(c: float, x: float, n: float) -> torch.Tensor:
    """
    Compute the derivative of a polynomial term c * x^n at point x.
    
    Args:
        c: coefficient of the term
        x: point at which to evaluate the derivative
        n: exponent of the term
    
    Returns:
        The value of the derivative at point x as a tensor
    """
    if n == 0:
        return torch.tensor(0.0)
    x = torch.tensor(x,dtype = torch.float32)

    out = c*n*torch.pow(x,n-1)

    return out