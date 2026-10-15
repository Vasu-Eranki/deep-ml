def add_two_lists(x:list[float],y:list[float])->list[float]:
	return [i+j for i,j in zip(x,y)]
def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	"""
	Perform RMSProp optimization update.
	
	Args:
		params: List of parameter values
		grads: List of gradients for each parameter
		cache: List of cache values (moving average of squared gradients)
		lr: Learning rate
		beta: Decay rate for moving average
		epsilon: Small constant for numerical stability
	
	Returns:
		Tuple of (updated_params, updated_cache)
	"""
	# Your code here
	
	squared_of_gradients = [(1-beta)*i**2 for i in grads]
	weighting_current_cache = [beta*i for i in cache]
	new_velocity = add_two_lists(weighting_current_cache,squared_of_gradients)
	square_root_of_velocity = [(i)**0.5+epsilon for i in new_velocity]
	params = [i - lr*(j/k) for i,j,k in zip(params,grads,square_root_of_velocity)]
	return params,new_velocity