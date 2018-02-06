import torch
import torch.nn.functional as nn
import gzip
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

with gzip.open('mnist_small.gz', 'r') as f:
    mnist_images, mnist_labels = cPickle.load(f)

## Set number of samples and number of X variables
N_samples, X_dim = mnist_images.shape 
## Mini batch size
mb_size = 100
## Fix dimensionality of latent variables (output layer)
Z_dim = 2
## Dimensionality of the hidden layer
h_dim = 128 
## Learning rate for stochastic gradient descent
lr = 1e-3

print mnist_images.shape
## Helper functions.
def mnist_mb(mb_size):
    """Sample batch of size mb_size from training data"""
    yield mnist_images[np.random.choice(N_samples, size=mb_size, replace=True),]
def init_weight(size):
    return Variable(torch.randn(*size) * (1. / np.sqrt(size[0] / 2.)), requires_grad=True)


## Initialize parameters of the encoder
Wxh = init_weight(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)
Whz_mu = init_weight(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)
Whz_var = init_weight(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)

## Initialize parameter of the decoder
Wzh = init_weight(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)
Whx = init_weight(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)

## Initilization for the run and run (controller) function
params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]
solver = torch.optim.Adagrad(params, lr=lr)
losses = [10**10]

## Encoder neural network, Q(Z|X)=N(z|mu,sigma) function
def Q(X):
    h = nn.relu(X.mm(Wxh) + bxh.repeat(X.size(0), 1))
    z_mu = h.mm(Whz_mu) + bhz_mu.repeat(h.size(0), 1)
    z_var = h.mm(Whz_var) + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var

## Sample from Z after applying reparameterization trick
def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps

## Initialize parameter of the decoder
Wzh = init_weight(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)
Whx = init_weight(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)

## Decoder neural network
def P(z):
    h = nn.relu(z.mm(Wzh) + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h.mm(Whx) + bhx.repeat(h.size(0), 1))
    return X

## Initilization for the run and run (controller) function
params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]
solver = torch.optim.Adagrad(params, lr=lr)
losses = [10**10]

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, (i+1))
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()
    

def run_ndim(num_iter,conv_check):
    conv=False
    
    for iter in range(num_iter):
        ## Load data.
        X = mnist_mb(mb_size=mb_size).next()
        X = Variable(torch.from_numpy(X))

        ## Forward propagate through network
        ## Encode X using Q network, sample using reparameterization trick, decode sample Z with P network
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)
        X_sample = P(z)
      
        ## Compute Loss of decoding, cross entropy between decoded data and input data
        recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
        ## Compute KL divergence between Q(Z|X)= N(z|mu,sig) and P(Z) = N(0,1)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        
        ## Compute total loss and evaluate convergence
        loss = recon_loss + kl_loss
        losses.append(loss.data.numpy()[0])
        if abs(losses[-1] - losses[-2]) < conv_check:
            conv = True
            break
        
        ## Backpropagation error towards input layer (and afterwards update weights in solver.step)
        loss.backward()

        solver.step()
        for p in params:
            p.grad.data.zero_()
        if(iter % 500 == 0):
            samples         = [sample_z(z_mu,z_var) for n in range(5)]
            decoded_samples_batch = [P(sam) for sam in samples] 
            print ('iteration: ', iter)
            plot_sam =decoded_samples_batch[0][0:4]
            plot_sam2 = plot_sam.data.numpy()
            
            plot_digits(plot_sam2,2)
        
        ## Now weights are updated for the mini batch and a new mini batch is chosen for num_iter
    z_M,z_S = Q(X)
    return losses, z_M,z_S, conv

run_ndim(1000,0.001)
