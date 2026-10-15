import math

def vector_multiplication(x:list[float],y:list[float])->float:
	multiplication = [i*j for i,j in zip(x,y)]
	return sum(multiplication)

def sigmoid(x:float)->float:
	result = 1+math.exp(-x)
	return 1/result

def mse_loss(x:list[float],y:list[int])->float:
	error = [(i-j)**2 for i,j in zip(x,y)]
	sse = sum(error)
	mse = sse/len(x)
	return mse

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	multiplication = [vector_multiplication(f,weights) for f in features]
	activations = [sigmoid(i+bias) for i in multiplication]
	probabilities = activations
	mse = mse_loss(probabilities,labels)
	rounded_prob = [round(i,4) for i in probabilities]
	return rounded_prob, round(mse,4)