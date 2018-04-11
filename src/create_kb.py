import random

size_dataset = 500000
for num in range(size_dataset):
	kb_i 	= np.zeros(1,5,5)
	var_a 	= random.randint(0,4)

	range_b = range(0,5)
	range_b = var_b.remove(var_a)
	print(range_b)
	var_b 	= random.choice(range_b)
	print(var_b)