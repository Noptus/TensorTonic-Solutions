def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """Returns the micro-averaged F1 score as a Python float rounded to four decimals."""
    tp = fp = fn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            tp += 1
        else:
            fp += 1
            fn += 1

    denominator = 2 * tp + fp + fn
    return round((2 * tp / denominator) if denominator else 0.0, 4)