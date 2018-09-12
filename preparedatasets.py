from collections import OrderedDict, defaultdict
import pandas as pd
from ptb import PTB
from utils import to_var, idx2word, expierment_name
import argparse
import csv
import torch
from torch.utils.data import DataLoader,Dataset
from multiprocessing import cpu_count
import numpy as np

class PandasDataset(Dataset):

    def __init__(self, csv_file):
        self.pandas_df = pd.read_csv(csv_file)

    def __len__(self):
    	return len(self.pandas_df)

    def __getitem__(self, index):
    	entry = self.pandas_df.iloc[index, :]
    	sen,attr = entry['ref'],entry['mr']
    	return sen, attr

def load_e2e(args):
	test=False
	splits = ['train']#, 'test'] 
	#Reading in text files of E2E database
	w_datasets = OrderedDict()
	for split in splits:
		w_datasets[split] = PTB(
		data_dir='e2e-dataset',
		split=split,
		create_data=args.create_data,
		max_sequence_length=args.max_sequence_length,
		min_occ=args.min_occ
		)

	#Reading in attributes of E2E data
	predicate_dict     = defaultdict(set)
	#predicate_dict_dev = defaultdict(set)
	df      = pd.read_csv('./e2e-dataset/trainset.csv', delimiter=',')
	tuples  = [tuple(x) for x in df.values]
	for t in tuples:
	    for r in t[0].split(','):
	        r_ind1 = r.index('[')
	        r_ind2 = r.index(']')
	        rel = r[0:r_ind1].strip()
	        rel_val = r[r_ind1+1:r_ind2]
	        predicate_dict[rel].add(rel_val)

	"""
	df_dev = pd.read_csv('./e2e-dataset/devset.csv' , delimiter=',')
	dev_tuples  = [tuple(x) for x in df_dev.values]
	for t in dev_tuples:
	    for r in t[0].split(','):
	        r_ind1 = r.index('[')
	        r_ind2 = r.index(']')
	        rel = r[0:r_ind1].strip()
	        rel_val = r[r_ind1+1:r_ind2]
	        predicate_dict[rel].add(rel_val)
	#print('preddict',predicate_dict_dev)
	print('preddict',predicate_dict)
	"""
	rel_lens = [len(predicate_dict[p]) for p in predicate_dict.keys()]

	print('kb')
	rel_list = list(predicate_dict.keys())
	print('rl')
	print(rel_list)
	rel_val_list = list(predicate_dict.values())
	X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.bool)
	X_test = np.zeros((len(dev_tuples), sum(rel_lens)), dtype=np.bool)

	for i, tup in enumerate(tuples):
		for relation in tup[0].split(',')[1:]:
			rel_name = relation[0:relation.index('[')].strip()
			rel_value= relation[relation.index('[')+1:-1].strip()
			name_ind = rel_list.index(rel_name)
			value_ind= list(predicate_dict[rel_name]).index(rel_value)
			j = sum(rel_lens[0:name_ind]) + value_ind
			X[i,j] = 1    
	"""
	for i, tup in enumerate(dev_tuples):
		for relation in tup[0].split(',')[1:]:
			rel_name = relation[0:relation.index('[')].strip()
			rel_value= relation[relation.index('[')+1:-1].strip()
			name_ind = rel_list.index(rel_name)
			value_ind= list(predicate_dict[rel_name]).index(rel_value)
			j = sum(rel_lens[0:name_ind]) + value_ind
			X_test[i,j] = 1
	"""
	return X,X_test
def main(args):
	test=False
	splits = ['train', 'test'] 
	#Reading in text files of E2E database
	w_datasets = OrderedDict()
	for split in splits:
		w_datasets[split] = PTB(
		data_dir='e2e-dataset',
		split=split,
		create_data=args.create_data,
		max_sequence_length=args.max_sequence_length,
		min_occ=args.min_occ
		)

	#Reading in attributes of E2E data
	predicate_dict     = defaultdict(set)
	#predicate_dict_dev = defaultdict(set)
	df      = pd.read_csv('./e2e-dataset/trainset.csv', delimiter=',')
	tuples  = [tuple(x) for x in df.values]
	for t in tuples:
	    for r in t[0].split(','):
	        r_ind1 = r.index('[')
	        r_ind2 = r.index(']')
	        rel = r[0:r_ind1].strip()
	        rel_val = r[r_ind1+1:r_ind2]
	        predicate_dict[rel].add(rel_val)

	df_dev = pd.read_csv('./e2e-dataset/devset.csv' , delimiter=',')
	dev_tuples  = [tuple(x) for x in df_dev.values]
	for t in dev_tuples:
	    for r in t[0].split(','):
	        r_ind1 = r.index('[')
	        r_ind2 = r.index(']')
	        rel = r[0:r_ind1].strip()
	        rel_val = r[r_ind1+1:r_ind2]
	        predicate_dict[rel].add(rel_val)
	#print('preddict',predicate_dict_dev)
	print('preddict',predicate_dict)

	rel_lens = [len(predicate_dict[p]) for p in predicate_dict.keys()]

	print('kb')
	rel_list = list(predicate_dict.keys())
	print('rl')
	print(rel_list)
	rel_val_list = list(predicate_dict.values())
	X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.bool)
	X_test = np.zeros((len(dev_tuples), sum(rel_lens)), dtype=np.bool)

	for i, tup in enumerate(tuples):
		for relation in tup[0].split(',')[1:]:
			rel_name = relation[0:relation.index('[')].strip()
			rel_value= relation[relation.index('[')+1:-1].strip()
			name_ind = rel_list.index(rel_name)
			value_ind= list(predicate_dict[rel_name]).index(rel_value)
			j = sum(rel_lens[0:name_ind]) + value_ind
			X[i,j] = 1    
	for i, tup in enumerate(dev_tuples):
		for relation in tup[0].split(',')[1:]:
			rel_name = relation[0:relation.index('[')].strip()
			rel_value= relation[relation.index('[')+1:-1].strip()
			name_ind = rel_list.index(rel_name)
			value_ind= list(predicate_dict[rel_name]).index(rel_value)
			j = sum(rel_lens[0:name_ind]) + value_ind
			X_test[i,j] = 1

	print('Xtest crated... shape:',X_test.shape)
	for i in range(5):
		print(tuples[i][0])
		ind_list = []
		for ind,a in enumerate(X[i]):
			if a ==True:
				ind_list.append(ind)#IS THIS WORKING???
		print(ind_list)

	pandata = PandasDataset(csv_file='./e2e-dataset/trainset.csv')
	panda_splits = ['a','b']

	for epoch in range(args.epochs):
	    #print(datasets['train'].shape)
	    for split in splits:#panda_splits:
	        pandata_loader = DataLoader(
	            dataset=pandata,
	            batch_size=args.batch_size,
	            shuffle=split=='train',
	            num_workers=cpu_count(),
	            pin_memory=torch.cuda.is_available()
	        )

	for epoch in range(args.epochs):
		for split in splits:
		    data_loader = DataLoader(
		        dataset=w_datasets[split],
		        batch_size=args.batch_size,
		        shuffle=split=='train',
		        num_workers=cpu_count(),
		        pin_memory=torch.cuda.is_available()
		    )

	for iteration, batch in enumerate(data_loader):
		if(iteration==0):
			a=1
			#print(batch)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--create_data', action='store_true')
	parser.add_argument('--max_sequence_length', type=int, default=60)
	parser.add_argument('--min_occ', type=int, default=1)
	parser.add_argument('--test', action='store_true')

	parser.add_argument('-ep', '--epochs', type=int, default=10)
	parser.add_argument('-bs', '--batch_size', type=int, default=32)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

	parser.add_argument('-eb', '--embedding_size', type=int, default=300)
	parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
	parser.add_argument('-hs', '--hidden_size', type=int, default=256)
	parser.add_argument('-nl', '--num_layers', type=int, default=1)
	parser.add_argument('-bi', '--bidirectional', action='store_true')
	parser.add_argument('-ls', '--latent_size', type=int, default=16)
	parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)


	parser.add_argument('-a','--alpha', type=float, default=0.0001)
	parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
	parser.add_argument('-k', '--k', type=float, default=0.0025)
	parser.add_argument('-x0', '--x0', type=int, default=2500)

	parser.add_argument('-v','--print_every', type=int, default=50)
	parser.add_argument('-tb','--tensorboard_logging', action='store_true')
	parser.add_argument('-log','--logdir', type=str, default='logs')
	parser.add_argument('-bin','--save_model_path', type=str, default='bin')

	args = parser.parse_args()

	args.rnn_type = args.rnn_type.lower()
	args.anneal_function = args.anneal_function.lower()

	assert args.rnn_type in ['rnn', 'lstm', 'gru']
	assert args.anneal_function in ['logistic', 'linear']
	print(load_e2e(args))
	main(args)
