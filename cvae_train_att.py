import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import pandas as pd
from ptb import PTB, PTBlabel
from utils import to_var, idx2word, expierment_name
from cvae_model_att import AttributeCVAE

def load_e2e(args):
    splits = ['train', 'valid'] 
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

    #Order both keys and items in dictionary for consistensy
    od = OrderedDict(sorted(predicate_dict.items()))
    for key in od.keys():
        od[key] = sorted(od[key])
    predicate_dict = od

    #print('preddict',predicate_dict_dev)
    rel_lens = [len(predicate_dict[p]) for p in predicate_dict.keys()]

    rel_list = list(predicate_dict.keys())
    rel_val_list = list(predicate_dict.values())
    X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.int)
    #X_test = np.zeros((len(dev_tuples), sum(rel_lens)), dtype=np.bool)

    for i, tup in enumerate(tuples):
        for relation in tup[0].split(','):
            rel_name = relation[0:relation.index('[')].strip()
            rel_value= relation[relation.index('[')+1:-1].strip()
            name_ind = rel_list.index(rel_name)
            value_ind= list(predicate_dict[rel_name]).index(rel_value)
            j = sum(rel_lens[0:name_ind]) + value_ind
            X[i,j] = 1.
        y_datasets = OrderedDict()
        #Same size as test set
        split_num = len(w_datasets['valid'])#is 4672
    y_datasets['train'] = X[0:-split_num]
    y_datasets['valid'] = X[-split_num:]
    assert (len(y_datasets['train']) ==len(y_datasets['train']))
    assert (len(w_datasets['valid'])== len(w_datasets['valid']))

    for split in splits:
        w_datasets[split] = PTBlabel(
        data_dir='e2e-dataset',
        split=split,
        labels=y_datasets[split],
        create_data=args.create_data,
        max_sequence_length=args.max_sequence_length,
        min_occ=args.min_occ
        )
    return w_datasets, y_datasets


def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] #+ (['test'] if args.test else [])

    #Just a random choice, change for real model !!!!!!!!!!!!!!
    """
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )
    """
    w_datasets,y_datasets = load_e2e(args)
    print((y_datasets[splits[0]].shape[1]))
    label_sequence_len = y_datasets[splits[0]].shape[1]

    model = AttributeCVAE(
        vocab_size=w_datasets['train'].vocab_size,
        sos_idx=w_datasets['train'].sos_idx,
        eos_idx=w_datasets['train'].eos_idx,
        pad_idx=w_datasets['train'].pad_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        label_sequence_len = label_sequence_len,
        bidirectional=args.bidirectional
        )

    if torch.cuda.is_available():
        model = model.cuda()

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join('./',args.logdir, expierment_name(args,ts)))
        writer.add_text("model_cvae", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join('./',args.save_model_path,'CVAE', ts)
    os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    BCE = torch.nn.BCELoss(size_average=False)

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target#[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp#.view(-1, logp.size(1))
        
        BCE_loss = BCE(logp,target)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return BCE_loss, KL_loss, KL_weight
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):
        print('ep',epoch)
        for split in splits:
            print(split)
            data_loader = DataLoader(
                dataset=w_datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)
 
            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):
                if torch.max(batch['input'])> 1683: #[33,36,72,73,83,129,157,158,165,177,181,201,274,352,459]:
                    continue
                batch_size = batch['input'].size(0)
                batch['labels']=batch['labels'].float()

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass

                logp, mean, logv, z = model(batch['input'],batch['labels'], batch['length'])

                # loss calculation
                BCE_loss, KL_loss, KL_weight = loss_fn(logp, batch['labels'],#batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (BCE_loss + KL_weight * KL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1


                # bookkeepeing
		# Avoid the .cat error !!!
                #print(loss.data)
                #print(tracker['ELBO'])
                loss_data = torch.cuda.FloatTensor([loss.data.item()]) if torch.cuda.is_available() else torch.tensor([loss.data.item()])
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss_data)) #Orig: tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data),1)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO"%split.upper(), loss.data[0], epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/BCE Loss"%split.upper(), BCE_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, BCE-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.data[0], BCE_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))
                
                #split = 'invalid' #JUST TO DEBUG!!!

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=w_datasets['train'].get_i2w(), pad_idx=w_datasets['train'].pad_idx) #ERROR HERE!!!
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('./dumps', ts)):
                    os.makedirs('./dumps/'+ts)
                with open(os.path.join('./dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train' and epoch %10 ==0:
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='e2e-dataset')
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

    main(args)
