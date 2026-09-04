import numpy as np

def rmsprop_step(w: list, g: list, s: list, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8) -> tuple[list, list]:
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    new_s = beta * s + (1 - beta) * (g ** 2)
    new_w = w - lr * g / (np.sqrt(new_s) + eps)

    return new_w.tolist(), new_s.tolist()