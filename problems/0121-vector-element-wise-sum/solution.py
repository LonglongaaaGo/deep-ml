def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    if len(a) != len(b): return -1
    out_list = []
    for a_,b_ in zip(a,b):
        out_list.append(a_+b_)
    
    return out_list