import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sum(x * p))