import math

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	# Your code here
	# (e^x - e^-x)/(e^x + e^-x)
	positive = math.exp(x)
	negative = math.exp(-x)
	numerator = positive -negative 
	denominator = positive + negative 
	return numerator/denominator 