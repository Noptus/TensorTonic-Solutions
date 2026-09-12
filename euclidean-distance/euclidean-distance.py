import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """Returns the Euclidean distance as a Python float."""
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.sqrt(np.sum((x - y) ** 2)))