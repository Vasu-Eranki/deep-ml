import math

def softmax(scores: list[float]) -> list[float]:
    max_value =max(scores)
    scores = [math.exp(i-max_value) for i in scores]
    sum_value = sum(scores)
    scores = [i/sum_value for i in scores]
    return scores