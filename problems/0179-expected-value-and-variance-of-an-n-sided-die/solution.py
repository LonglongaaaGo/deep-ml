def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    
    exp_ = 0
    for i in range(1,n+1):
        exp_+= i/n
    var_ = 0
    for i in range(1,n+1):
        var_ += 1/n* (i-exp_)*(i-exp_)
    
    return exp_,var_