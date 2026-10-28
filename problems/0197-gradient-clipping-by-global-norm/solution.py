def square(x:list[float])->float:
	return sum([i**2 for i in x])
def scale_list(x:list[float],scale:float)->float:
	return [i*scale for i in x]
def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
	"""
	Clip gradients by global norm.
	
	Args:
		gradients: List of gradient arrays
		max_norm: Maximum allowed global norm
	
	Returns:
		List of clipped gradient arrays
	"""
	# Your code here
	norm = sum([square(i) for i in gradients])**0.5
	if(norm>max_norm):
		scale = max_norm/norm
		gradients = [scale_list(i,scale) for i in gradients]
	return gradients	