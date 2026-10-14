import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    maximum_value  = max(scores)
    scores = [math.exp(i-maximum_value) for i in scores]
    total_sum = sum(scores)
    scores = [i/total_sum for i in scores]
    return scores