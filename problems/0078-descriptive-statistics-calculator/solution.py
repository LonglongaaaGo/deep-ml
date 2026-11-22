import numpy as np
from collections import Counter
def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    arr = np.asarray(data,dtype=float)
    if arr.size == 0:
        raise ValueError(" error ")
    counts = Counter(arr.tolist())
    max_freq = max(counts.values())
    mode_val = int(min(val for val,count in counts.items() if count == max_freq))

    q25,q50,q75 = np.percentile(arr,[25,50,75],method="linear")

    variance_val = float(np.var(arr,ddof = 0))
    std_dev_val = float(np.std(arr,ddof=0))
    return {

        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "mode": mode_val,
        "variance": variance_val,
        "standard_deviation": std_dev_val,
        "25th_percentile": float(q25),
        "50th_percentile": float(q50),
        "75th_percentile": float(q75),
        "interquartile_range": float(q75 - q25),

    }

    
