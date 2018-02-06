import random
f = open('dfa.dat', 'w')

size_dataset = 500000

for num in range(size_dataset):
	f.write('a')
	next_char = random.choice(['d','b'])
	bs = 0
	while (not next_char == 'c'):
		f.write(next_char)
		if(next_char=='d'):
			next_char = random.choice(['c','b'])
		if(next_char == 'b' and bs<5):
			bs +=1
			next_char = random.choice(['c','b'])
		if (bs ==5):
			next_char = 'c'
		#print(next_char)
	f.write(next_char+'\n')
f.write('abc')
f.close()