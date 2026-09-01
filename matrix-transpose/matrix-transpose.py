import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    rows = len(A)
    cols = len(A[0])

    result = np.empty((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j, i] = A[i][j]

    return result