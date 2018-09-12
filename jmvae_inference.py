import os
import json
import torch
import argparse
import random
import pandas as pd
from jmvae_model import SentenceJMVAE
from utils import to_var, idx2word, interpolate
from jmvae_train import load_e2e
from torch.utils.data import DataLoader,Dataset
from multiprocessing import cpu_count
from collections import defaultdict, OrderedDict
import numpy as np
def main(args):



    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceJMVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        label_sequence_len=args.label_sequence_len
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    print('summary')
    model.load_state_dict(torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage))
    #print("Model loaded from s%"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()


    #Create decoding dict for attributes
    predicate_dict     = defaultdict(set)

    #Load in attribute types from trainset
    df      = pd.read_csv('./e2e-dataset/trainset.csv', delimiter=',')
    tuples  = [tuple(x) for x in df.values]
    #Parse all  the attribute inputs
    for t in tuples:
        for r in t[0].split(','):
            r_ind1 = r.index('[')
            r_ind2 = r.index(']')
            rel = r[0:r_ind1].strip()
            rel_val = r[r_ind1+1:r_ind2]
            predicate_dict[rel].add(rel_val)

    #Sort attribute inputs for consistensy for each run 
    od = OrderedDict(sorted(predicate_dict.items()))
    for key in od.keys():
        od[key] = sorted(od[key])
    predicate_dict = od
    rel_lens = [len(predicate_dict[p]) for p in predicate_dict.keys()]
    print('lrl', len(rel_lens))
    rel_list = list(predicate_dict.keys())
    rel_val_list = list(predicate_dict.values())    

    X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.int)
    #Generate input in matrix form
    int_to_rel = defaultdict()
    for i, tup in enumerate(tuples):
        for relation in tup[0].split(','):
            rel_name = relation[0:relation.index('[')].strip()
            rel_value= relation[relation.index('[')+1:-1].strip()
            name_ind = rel_list.index(rel_name)
            value_ind= list(predicate_dict[rel_name]).index(rel_value)
            j = sum(rel_lens[0:name_ind]) + value_ind
            #print(relation,j)
            int_to_rel[j] = relation
            X[i,j] = 1.

    ii = 22
    print(tuples[ii])
    print(X[ii])
    indices_att = [int_to_rel[i] for i, x in enumerate(X[ii]) if x == 1]
    print(indices_att)
    y_datasets = OrderedDict()
    print('----------DECODINGS----------')

    w_datasets,y_datasets = load_e2e(False,args.max_sequence_length,1)
    print('a')
    batch_size = 10
    data_loader = DataLoader(
        dataset=w_datasets['test'],#y_datasets[split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    batch_num = random.randint(0,len(data_loader)-1)
    max_max_word = 0
    false_neg_list = {'label':[],'joint':[],'sentence':[]}
    false_pos_list = {'label':[],'joint':[],'sentence':[]}
    acc_list = {'label':[],'joint':[],'sentence':[]}
    loss_list = {'label':[],'joint':[],'sentence':[]}
    perfect = {'label':[],'joint':[],'sentence':[]} 

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=w_datasets['train'].pad_idx)
    BCE = torch.nn.BCELoss(size_average=False)
    def loss_fn_plus(logp, logp2, target,target2, length, mean, logv, mean_w, logv_w, mean_y, logv_y, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        #Cross entropy loss
        BCE_loss = BCE(logp2,target2)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        return NLL_loss, BCE_loss, KL_loss
    for iteration, batch in enumerate(data_loader):
        batch_size = len(batch['input'])
        #if not iteration==batch_num:
        #    continue

        #Make sure all word indeces within range of trained model
        if torch.max(batch['input'])> 1683: #[33,36,72,73,83,129,157,158,165,177,181,201,274,352,459]:
            print(iteration)
            continue
        batch['labels']=batch['labels'].float()
        #for k, v in batch.items():
        #    if torch.is_tensor(v):
        #        batch[k] = to_var(v)
        sorted_lengths, sorted_idx = torch.sort(batch['length'], descending=True)
        input_sequence = batch['input'][sorted_idx]
        #print(input_sequence)
        #import pdb; pdb.set_trace()
        if torch.max(input_sequence) > max_max_word:
            max_max_word = torch.max(input_sequence)
        print(max_max_word)
        #print(iteration,torch.max(input_sequence))
        input_embedding = model.embedding(input_sequence)
        label_sequence = batch['labels'][sorted_idx]
        param_y = model.encode_y(label_sequence)
        param_joint = model.encode_joint(input_embedding,label_sequence, sorted_lengths)
        param_w = model.encode_w(input_embedding, sorted_lengths)

        _,reversed_idx = torch.sort(sorted_idx)

        params = [(param_y,'label'),(param_joint,'joint'),(param_w,'sentence')]
        print_stuff = False
        if iteration == batch_num:
            print_stuff = True
        for param,name in params:
            if print_stuff:
                print('----------Reconstructions from '+name+' data----------')
            mu_i,sig_i= param
            z_w = model.sample_z(batch_size,mu_i,sig_i)
            #_, y_decoded = model.decode_joint(z_w,input_embedding,sorted_lengths,sorted_idx)
            y_decoded = model.decode_to_y(z_w,sorted_lengths,sorted_idx)
            samples_w, z_w2 = model.inference(n=args.num_samples, z=z_w)
            inp_samp = idx2word(input_sequence, i2w=i2w, pad_idx=w2i['<pad>'])
            dec_samp = idx2word(samples_w, i2w=i2w, pad_idx=w2i['<pad>'])
            for iter in zip(inp_samp,dec_samp,label_sequence,y_decoded):
                if print_stuff:
                    print('True',iter[0],'\n','Pred',iter[1])

                true_att = [int_to_rel[i] for i,x in enumerate(iter[2].round()) if x == 1.]
                pred_att = [int_to_rel[i] for i,x in enumerate(iter[3].round()) if x == 1.]
                if print_stuff:
                    print('True',true_att,'\nPred',pred_att)
                mistakes = 0
                false_pos = 0
                false_neg = 0
                for t,p in zip(iter[2].round(),iter[3].round()):
                    if t!=p:
                        mistakes +=1
                        if p==0.:
                            false_neg+=1
                        elif p==1.:
                            false_pos+=1
                if print_stuff:
                    print('Mistakes:',mistakes )
                    print('Accuracy:',(len(iter[2])-mistakes)/len(iter[2]))
                    print('False pos:',false_pos,'\tFalse neg:',false_neg,'\n')
                false_neg_list[name].append(false_pos)
                false_pos_list[name].append(false_neg)
                acc_list[name].append((len(iter[2])-mistakes)/len(iter[2]))


            #logp, logp2, mean, logv, z, mean_w, logv_w, mean_y, logv_y = model(batch['input'],batch['labels'], batch['length'])
            loss_current = [0,0,0]#loss_fn_plus(logp, logp2, batch['target'], batch['labels'], 
                #batch['length'], mean, logv, mean_w, logv_w, mean_y, logv_y, 'logistic', 1000, 0.0025, 2500)

            loss_list[name].append(loss_current)
        print_stuff = False

            #NLL_loss, BCE_loss, KL_loss, KL_loss_w, KL_loss_y, KL_weight
                #print('Attributes')
    for _, name in params:
        print(name+':')     
        print('mmw',max_max_word)
        print('avg false neg',sum(false_neg_list[name])/len(false_neg_list[name]))
        print('avg false pos',sum(false_pos_list[name])/len(false_pos_list[name]))
        print('avg accuracy', sum(acc_list[name])/len(acc_list[name]))
        print('avg NLL', sum([x[0] for x in loss_list[name]])/len(loss_list[name]))
        print('avg BCE', sum([x[1] for x in loss_list[name]])/len(loss_list[name]))
        print('avg KL joint div', sum([x[2] for x in loss_list[name]])/len(loss_list[name]))

    samples, z = model.inference(n=args.num_samples)
    print()
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    #Make uncoupled label decoder 
    #
    #
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='e2e-dataset')#default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-lsl', '--label_sequence_len', type=int, default=79)

    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
