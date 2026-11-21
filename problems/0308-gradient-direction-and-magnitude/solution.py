import torch

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient (float)
        - direction: Unit vector (torch.Tensor) in direction of steepest ascent
        - descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
    """
    gradient = torch.as_tensor(gradient,dtype=torch.float32)
    
    magnitude = torch.linalg.vector_norm(gradient,ord=2).item()
    if magnitude == 0:
        direction = torch.zeros_like(gradient)
    else:
        direction = gradient/magnitude
    descent_direction = -direction

    return {"magnitude":magnitude,"direction":direction,"descent_direction":descent_direction}



