import numpy as np

def entropy_node(y: list[int]) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))