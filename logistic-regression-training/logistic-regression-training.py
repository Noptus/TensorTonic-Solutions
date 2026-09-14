def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000,
) -> tuple[np.ndarray, float]:
    """Returns the trained weights and bias as (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(steps):
        logits = X @ w + b
        p = _sigmoid(logits)

        error = p - y
        dw = (X.T @ error) / n_samples
        db = float(np.mean(error))

        w -= lr * dw
        b -= lr * db

    return w, b