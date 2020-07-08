from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import numpy as np
import h5py
import time

import torch_utils
import data_utils

# global params

parser = argparse.ArgumentParser(description='DANet')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: True)')
parser.add_argument('--seed', type=int, default=20170220,
                    help='random seed (default: 20170220)')
parser.add_argument('--infeat-dim', type=int, default=129,
                    help='dimension of the input feature (default: 129)')
parser.add_argument('--outfeat-dim', type=int, default=20,
                    help='dimension of the embedding (default: 20)')
parser.add_argument('--threshold', type=float, default=0.9,
                    help='the weight threshold (default: 0.9)')
parser.add_argument('--seq-len', type=int, default=100,
                    help='length of the sequence (default: 100)')
parser.add_argument('--log-step', type=int, default=100,
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--num-layers', type=int, default=4,
                    help='number of stacked RNN layers (default: 1)')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional RNN layers (default: True)')
parser.add_argument('--val-save', type=str,  default='model.pt',
                    help='path to save the best model')

args, _ = parser.parse_known_args()
args.cuda = args.cuda and torch.cuda.is_available()
args.num_direction = int(args.bidirectional)+1

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} 
else:
    kwargs = {}

# training and validation datast path
training_data_path = 'your_path_to_training_set'
validation_data_path = 'your_path_to_validation_set'

# define data loaders

train_loader = DataLoader(data_utils.WSJDataset(training_data_path), 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          **kwargs)

validation_loader = DataLoader(data_utils.WSJDataset(validation_data_path), 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          **kwargs)
# define model

class DANet(nn.Module):
    def __init__(self):
        super(DANet, self).__init__()
        
        self.rnn = torch_utils.MultiRNN('LSTM', args.infeat_dim, 300, 
                                           num_layers=args.num_layers, 
                                           bidirectional=args.bidirectional)
        self.FC = torch_utils.FCLayer(600, args.infeat_dim*args.outfeat_dim, nonlinearity='tanh')
        
        self.infeat_dim = args.infeat_dim
        self.outfeat_dim = args.outfeat_dim
        self.eps = 1e-8
        
    def forward(self, input, ibm, weight, hidden):
        """
        input: the input feature; 
            shape: (B, T, F)
            
        ibm: the ideal binary mask used for calculating the 
            ideal attractors; 
            shape: (B, T*F, nspk)
            
        weight: the binary energy threshold matrix for masking 
            out T-F bins; 
            shape: (B, T*F, 1)
            
        hidden: the initial hidden state in the LSTM layers.
        """
        
        seq_len = input.size(1)
        
        # generate the embeddings (V) by the LSTM layers
        LSTM_output, hidden = self.rnn(input, hidden)
        LSTM_output = LSTM_output.contiguous().view(-1, LSTM_output.size(2))  # B*T, H 
        V = self.FC(LSTM_output)  # B*T, F*K
        V = V.view(-1, seq_len*self.infeat_dim, self.outfeat_dim)  # B, T*F, K
        
        # calculate the ideal attractors
        # first calculate the source assignment matrix Y
        Y = ibm * weight.expand_as(ibm) # B, T*F, nspk
        
        # attractors are the weighted average of the embeddings
        # calculated by V and Y
        V_Y = torch.bmm(torch.transpose(V, 1,2), Y)  # B, K, nspk
        sum_Y = torch.sum(Y, 1, keepdim=True).expand_as(V_Y)  # B, K, nspk
        attractor = V_Y / (sum_Y + self.eps)  # B, K, 2
        
        # calculate the distance bewteen embeddings and attractors
        # and generate the masks
        dist = V.bmm(attractor)  # B, T*F, nspk
        mask = F.softmax(dist, dim=2)  # B, T*F, nspk
        
        return mask, hidden
    
    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)
        
    
def objective(mixture, wfm, estimated_mask):
    """
    MSE as the training objective. The mask estimation loss is calculated.
    You can also change it into the spectrogram estimation loss, which is 
    to calculate the MSE between the clean source spectrograms and the 
    masked mixture spectrograms.
    
    mixture: the spectrogram of the mixture;
        shape: (B, T, F)
        
    wfm: the target masks, which are the wiener-filter like masks here;
        shape: (B, T*F, nspk)
    
    estimated_mask: the estimated masks generated by the network;
        shape: (B, T*F, nspk)
    """
    
    loss = mixture.expand(mixture.size(0), mixture.size(1), wfm.size(2)) * (wfm - estimated_mask)
    loss = loss.view(-1, loss.size(1)*loss.size(2))
    
    return torch.mean(torch.sum(torch.pow(loss, 2), 1))
    
# define the model and the optimizer
model = DANet()
if args.cuda:
    model.cuda()

current_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler  = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
scheduler.step()

# function for training and validation

def train(epoch):
    start_time = time.time()
    model.train()
    train_loss = 0.
    
    # data loading
    # see data_utils.py for dataloader details
    for batch_idx, data in enumerate(train_loader):
        # batch_infeat is the input feature
        batch_infeat = Variable(data[0]).contiguous()
        
        # wiener-filter like mask as the training target
        batch_wfm = Variable(data[1]).contiguous()
        
        # spectrogram of mixture, used in objective
        batch_mix = Variable(data[2]).contiguous()
        
        # ideal binary mask as the ideal source assignment
        # used during the calculation of attractors
        batch_ibm = Variable(data[3]).contiguous()
        
        # energy threshold matrix calculated from the mixture spectrogram
        batch_weight = Variable(data[4]).contiguous()
        
        if args.cuda:
            batch_infeat = batch_infeat.cuda()  # B, T, F
            batch_wfm = batch_wfm.cuda()  # B, T*F, nspk
            batch_mix = batch_mix.cuda()  # B, T, F
            batch_ibm = batch_ibm.cuda()  # B, T*F, nspk
            batch_weight = batch_weight.cuda()  # B, T*F, 1
        
        # training
        hidden = model.init_hidden(batch_infeat.size(0))
        optimizer.zero_grad()
        estimated_mask, hidden = model(batch_infeat, batch_ibm, batch_weight, hidden)
        
        loss = objective(batch_mix, batch_wfm, estimated_mask)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        
        # output logs
        if (batch_idx+1) % args.log_step == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(train_loader),
                elapsed * 1000 / (batch_idx+1), train_loss / (batch_idx+1)))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss
        
def validate(epoch):
    start_time = time.time()
    model.eval()
    validation_loss = 0.
    
    # data loading
    for batch_idx, data in enumerate(validation_loader):
        batch_infeat = Variable(data[0]).contiguous()
        batch_wfm = Variable(data[1]).contiguous()
        batch_mix = Variable(data[2]).contiguous()
        batch_ibm = Variable(data[3]).contiguous()
        batch_weight = Variable(data[4]).contiguous()
        
        if args.cuda:
            batch_infeat = batch_infeat.cuda()
            batch_wfm = batch_wfm.cuda()
            batch_mix = batch_mix.cuda()
            batch_ibm = batch_ibm.cuda()
            batch_weight = batch_weight.cuda()
        
        # mask estimation
        with torch.no_grad():
            hidden = model.init_hidden(batch_infeat.size(0))
            estimated_mask, hidden = model(batch_infeat, batch_ibm, batch_weight, hidden)
        
            loss = objective(batch_mix, batch_wfm, estimated_mask)
            validation_loss += loss.data.item()
    
    validation_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss

# main function

training_loss = []
validation_loss = []
decay_cnt = 0
for epoch in range(1, args.epochs + 1):
    model.cuda()
    training_loss.append(train(epoch))
    validation_loss.append(validate(epoch))
    if training_loss[-1] == np.min(training_loss):
        print('      Best training model found.')
        print('-' * 99)
    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(args.val_save, 'wb') as f:
            torch.save(model.cpu().state_dict(), f)
            print('      Best validation model found and saved.')
            print('-' * 99)
    decay_cnt += 1
    # lr decay
    if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
        scheduler.step()
        decay_cnt = 0
        print('      Learning rate decreased.')
        print('-' * 99)