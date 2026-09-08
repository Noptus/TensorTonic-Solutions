def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)

    precision = hits / k
    recall = hits / len(relevant)

    return [precision, recall]