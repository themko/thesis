import os
import torch
import torch.utils.data
import argparse
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab

def main(args):
	train_losses = []
	valid_losses = []

	with open(args.file_path,'r') as readfile:
		for line in readfile:
			line = line.strip(',')
			if(len(line.split()) ==0):
				continue
			if (line.split()[0] == 'TRAIN') and len(line.split())>6:
				train_losses.append([l.strip(',') for l in line.split()])#[3:4+1])
			if (line.split()[0] == 'VALID') and len(line.split())>6:
				valid_losses.append([l.strip(',') for l in line.split()])#[3:4+1])

	train_losses 	= np.array(train_losses)
	valid_losses 	= np.array(valid_losses)
	print(train_losses[0])
	print(len(train_losses[0]))
	train_loss_numbers = train_losses[:,range(4,len(train_losses[0]),2)]
	train_loss_names = train_losses[:,range(3,len(train_losses[0]),2)]
	val_loss_numbers = valid_losses[:,range(4,len(train_losses[0]),2)]
	val_loss_names = valid_losses[:,range(3,len(train_losses[0]),2)]
	#JMVAE case

	output_folder  = args.file_path+'_figs'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	loss_nums	= train_loss_numbers
	loss_nums = np.transpose(np.array(loss_nums))

	loss_names 	= train_loss_names

	val_loss_nums	= val_loss_numbers
	val_loss_nums	= np.transpose(np.array(val_loss_nums))
	val_loss_names 	= val_loss_names

	print(loss_nums[0])
	print(loss_names[0])
	print('\nTrain plots:')
	for current_losses,name in zip(loss_nums,loss_names[0]):
		current_loss = [float(loss) for loss in current_losses]
		print('start plot',name)
		plt.plot(current_loss,linewidth=1.1,color='b')
		plt.title('Training '+str(name))
		plt.xlabel('Epochs')
		plt.ylabel('Batch loss value')
		plt.xticks(np.arange(0,1.0,0.099)*len(current_loss),np.arange(0,110,10))
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.tick_params(top='off', right='off')
		plt.savefig(output_folder+'/train_'+str(name)+'.png')
		plt.close()
		#plt.show()

	print('\nTest plots:')
	for current_losses,name in zip(val_loss_nums,val_loss_names[0]):
		current_loss = [float(loss) for loss in current_losses]
		print('start plot',name)
		plt.plot(current_loss,linewidth=1.1,color='b')
		plt.title('Test '+ name)
		plt.xlabel('Epochs')
		plt.ylabel('Batch loss value')
		plt.xticks(np.arange(0,1.0,0.099)*len(current_loss),np.arange(0,110,10))
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.tick_params(top='off', right='off')
		plt.savefig(output_folder+'/valid_'+str(name)+'.png')
		plt.close()
	
if __name__ == '__main__':
    print('Hello')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str)
    args = parser.parse_args()
    main(args)
