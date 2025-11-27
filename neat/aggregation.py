import sys

from operator import mul

from neat.math_util import mean, median2

if sys.version_info[0] > 2:
    from functools import reduce

def product_aggregation(x): # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)

def sum_aggregation(x):
    return sum(x)

def max_aggregation(x):
    return max(x)

def min_aggregation(x):
    return min(x)

def maxabs_aggregation(x):
    return max(x, key=abs)

def median_aggregation(x):
    return median2(x)

def mean_aggregation(x):
    return mean(x)

class AggregationFunction:

    functions = {
        'product': product_aggregation,
        'sum': sum_aggregation,
        'max': max_aggregation,
        'min': min_aggregation,
        'maxabs': maxabs_aggregation,
        'median': median_aggregation,
        'mean': mean_aggregation,
    }

    def __init__(self, name):
        if name not in AggregationFunction.functions:
            raise ValueError(f"Agg function '{name}' is not defined.")
        self.name = name
        self.function = AggregationFunction.functions[name]

        
