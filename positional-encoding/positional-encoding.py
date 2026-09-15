import numpy as np

def positional_encoding(
    seq_len: int,
    d_model: int,
    base: float = 10000.0
) -> np.ndarray:
    """Returns a NumPy array of shape (seq_len, d_model)."""
    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]
    even_dims = np.arange(0, d_model, 2, dtype=float)

    angles = positions / (base ** (even_dims / d_model))

    pe = np.empty((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe