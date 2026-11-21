import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if norm_type == "l1":
        out = torch.linalg.vector_norm(arr, ord=1)
        # out = torch.linalg.vector_norm(arr, ord=1)
    elif norm_type == "l2":
        out = torch.linalg.vector_norm(arr, ord=2)
    elif norm_type == "linf":
        out = torch.linalg.vector_norm(arr, ord=float("inf"))
    elif norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError("Frobenius norm requires a 2D tensor.")
        out = torch.linalg.matrix_norm(arr, ord="fro")
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    return out.item()
