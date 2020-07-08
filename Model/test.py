from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np
import h5py
import time

import torch_utils
import data_utils

import librosa
from sklearn.cluster import KMeans

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
    
# STFT parameters
sr = 8000
nfft = 256
nhop = 64
nspk = 2

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
        
    def forward(self, input, hidden):
        """
        input: the input feature; 
            shape: (B, T, F)
            
        hidden: the initial hidden state in the LSTM layers.
        """
        
        seq_len = input.size(1)
        
        # generate the embeddings (V) by the LSTM layers
        LSTM_output, hidden = self.rnn(input, hidden)
        LSTM_output = LSTM_output.contiguous().view(-1, LSTM_output.size(2))  # B*T, H 
        V = self.FC(LSTM_output)  # B*T, F*K
        V = V.view(-1, seq_len*self.infeat_dim, self.outfeat_dim)  # B, T*F, K
                
        return V
    
    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)

# load model
model = DANet()
model.load_state_dict(torch.load('model.pt'))

if args.cuda:
    model.cuda()
model.eval()

# load mixture data
mix, _ = librosa.load('your_path_to_mixture_audio', sr=sr)

# STFT
mix_spec = librosa.stft(mix, nfft, nhop)  # F, T
mix_phase = np.angle(mix_spec)  # F, T
mix_spec = np.abs(mix_spec)  # F, T

# magnitude spectrogram in db scale
infeat = 20*np.log10(mix_spec.T)
infeat = np.asarray([infeat]*1)
# optional: normalize the input feature with your pre-calculated
# statistics of the training set

batch_infeat = Variable(torch.from_numpy(infeat)).contiguous()
if args.cuda:
    batch_infeat = batch_infeat.cuda()

with torch.no_grad():
    hidden = model.init_hidden(batch_infeat.size(0))
    embeddings = model(batch_infeat, hidden)
    
# estimate attractors via K-means
embeddings = embeddings[0].data.cpu().numpy()  # T*F, K
kmeans_model = KMeans(n_clusters=nspk, random_state=0).fit(embeddings.astype('float64')) 
attractor = kmeans_model.cluster_centers_  # nspk, K

# estimate masks
embeddings = torch.from_numpy(embeddings).float()  # T*F, K
attractor = torch.from_numpy(attractor.T).float()  # K, nspk
if args.cuda:
    embeddings = embeddings.cuda()
    attractor = attractor.cuda()

mask = F.softmax(torch.mm(embeddings, attractor), dim=1)  # T*F, nspk
mask = mask.data.cpu().numpy()

mask_1 = mask[:,0].reshape(-1, args.infeat_dim).T
mask_2 = mask[:,1].reshape(-1, args.infeat_dim).T

# masking the mixture magnitude spectrogram
s1_spec = (mix_spec * mask_1) * np.exp(1j*mix_phase)
s2_spec = (mix_spec * mask_2) * np.exp(1j*mix_phase)

# reconstruct waveforms
res_1 = librosa.istft(s1_spec, hop_length=nhop, win_length=nfft)
res_2 = librosa.istft(s2_spec, hop_length=nhop, win_length=nfft)

if len(res_1) < len(mix):
    # pad zero at the end
    res_1 = np.concatenate([res_1, np.zeros(len(mix)-len(res_1))])
    res_2 = np.concatenate([res_2, np.zeros(len(mix)-len(res_2))])
else:
    res_1 = res_1[:len(mix)]
    res_2 = res_2[:len(mix)]