import random
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('path', type=str,nargs='?', default='/dfa.dat',const='/dfa.dat',
                    help='Parameter indicating if currently retraining model or just running existing one')
args        = parser.parse_args()
path  = args.path
f = open(path, 'w')

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