
from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    counter = Counter(samples)
    n = sum(counter.values())
    prob = []
    for value,count in sorted(counter.items()):
        
        prob.append((value,count/n))


    return prob
