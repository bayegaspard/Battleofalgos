def exact_match(pred, truth):
    if not isinstance(pred, list) or not isinstance(truth, list):
        return False
    return set(pred) == set(truth)


def jaccard(pred, truth):
    if not isinstance(pred, list) or not isinstance(truth, list) or not pred or not truth:
        return 0.0
        
    pred = set(pred)
    truth = set(truth)

    return len(pred & truth) / len(pred | truth)


def evaluate(predictions, ground_truth):
    if not predictions:
        return {"accuracy": 0, "jaccard": 0}
        
    exact = []
    jacc = []

    for p, t in zip(predictions, ground_truth):
        exact.append(exact_match(p, t))
        jacc.append(jaccard(p, t))

    return {
        "accuracy": sum(exact) / len(exact) if exact else 0,
        "jaccard": sum(jacc) / len(jacc) if jacc else 0
    }
