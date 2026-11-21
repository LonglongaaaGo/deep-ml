import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Your code here
    x = torch.as_tensor(x,dtype=torch.float32)
    
    def cal_derivatives(coeffs):
        n = len(coeffs)
        sum_dev = 0
        sum_ = 0

        for idx in (range(n-1,-1,-1)):
            coeff = coeffs[n-1-idx]
            if idx>0:
                sum_dev+= coeff*idx* torch.pow(x,idx-1)
            sum_+=coeff* torch.pow(x,idx)

        return sum_,sum_dev


    g_sum,g_sum_dev = cal_derivatives(g_coeffs)
    h_sum,h_sum_dev = cal_derivatives(h_coeffs)

    out = (g_sum_dev*h_sum - g_sum*h_sum_dev)/(h_sum*h_sum)
    return out

