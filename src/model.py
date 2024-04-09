#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:38:20 2023

@author: kurmanbek
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pickle
import argparse






probs = np.array([0.1, 0.2, 0.5, 0.8, 1])
class PlaylistTrackDataset(Dataset):
    def __init__(self, playlists, track_count, mode = 'trn', transform=None, target_transform=None):
        
            
        self.playlists = playlists
        self.track_count = track_count
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode 

    def __len__(self):
        return len(self.playlists)

    def __getitem__(self, idx):
        track_indices = self.playlists[idx][0]
        p = np.random.choice(probs)
        if self.mode == 'trn':
            X_indices = []
            for idx in track_indices:
                if np.random.rand() <= p:
                    X_indices.append(idx)
            
            if len(X_indices) == 0:
                X_indices = track_indices            
        else:
            X_indices = track_indices
            
            
        X_track_coords = [[0]*len(X_indices), X_indices]
        X_ones = np.ones(len(X_indices))
        X_sparse_ts = torch.sparse_coo_tensor(X_track_coords, X_ones, [1, self.track_count], dtype=torch.float32)
        
        Y_track_coords = [[0]*len(track_indices), track_indices]
        Y_ones = np.ones(len(track_indices))
        Y_sparse_ts = torch.sparse_coo_tensor(Y_track_coords, Y_ones, [1, self.track_count], dtype=torch.float32)
        return X_sparse_ts.to_dense(), Y_sparse_ts.to_dense()
    
    
class PlaylistTitleDataset(Dataset):
    def __init__(self, playlists, track_count, mode = 'trn', transform=None, target_transform=None):
        self.playlists = playlists
        self.track_count = track_count
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode 

    def __len__(self):
        return len(self.playlists)

    def __getitem__(self, idx):
        track_indices = self.playlists[idx][0]
        title_indices = self.playlists[idx][2]
        
        X = torch.zeros(1, NUM_CHAR, SEQ_LEN)
        
        for i in range(SEQ_LEN):
            if title_indices[i] != -1:
                X[0][title_indices[i]][i] = 1
            
        
        p = np.random.choice(probs)
        if self.mode == 'trn':
            Y_indices = []
            for idx in track_indices:
                if np.random.rand() <= p:
                    Y_indices.append(idx)
            
            if len(Y_indices) == 0:
                Y_indices = track_indices            
        else:
            Y_indices = track_indices
        
        Y_track_coords = [[0]*len(Y_indices), Y_indices]
        Y_ones = np.ones(len(Y_indices))
        Y_sparse_ts = torch.sparse_coo_tensor(Y_track_coords, Y_ones, [1, self.track_count], dtype=torch.float32)
        return X, torch.squeeze(Y_sparse_ts.to_dense(), dim=1)




H_DIM = 32
M_DIM = 2*H_DIM
L_DIM = 32

N_FILTERS = 20

# Create auto-encoder model
class TrackAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_encoder = nn.Sequential(
            nn.Linear(track_count, M_DIM),
            nn.Sigmoid(),
            nn.Linear(M_DIM, H_DIM),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )
        
        self.linear_decoder= nn.Sequential(
            nn.Linear(H_DIM, M_DIM),
            nn.Sigmoid(),
            nn.Linear(M_DIM, track_count)
        )

    def forward(self, x):
        encoded = self.linear_encoder(x)
        decoded = self.linear_decoder(encoded)
        return decoded


class TitleAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, N_FILTERS, (NUM_CHAR, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, SEQ_LEN-3)),
            nn.Flatten(),
            nn.Unflatten(1, (1, N_FILTERS)),
            nn.Dropout(0.2)
            )
        
        self.linear_decoder= nn.Sequential(
            nn.Linear(N_FILTERS, M_DIM),
            nn.Sigmoid(),
            nn.Linear(M_DIM, track_count)
        )
        

    def forward(self, x):
        conv = self.conv1(x)
        #print('Conv shape', conv.size())
        decoded = self.linear_decoder(conv)
        return decoded




device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



def get_model(training_input):
    if training_input == 'title':
        return TitleAutoEncoder().to(device)
    return TrackAutoEncoder().to(device)

def get_trn_vld_dataloaders(training_input, trn_playlists, vld_playlists):
    
    if training_input == 'title':
        trn_dataset = PlaylistTitleDataset(trn_playlists, track_count, mode='trn')
        vld_dataset = PlaylistTitleDataset(vld_playlists, track_count, mode='vld')
    elif training_input == 'track':
        trn_dataset = PlaylistTrackDataset(trn_playlists, track_count, mode='trn')
        vld_dataset = PlaylistTrackDataset(vld_playlists, track_count, mode='vld')
    else:
        raise ValueError('Incorrect training input')
    
    trn_dataloader = DataLoader(dataset=trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
    vld_dataloader = DataLoader(dataset=vld_dataset, batch_size = BATCH_SIZE, shuffle=False)
             
                
    return trn_dataloader, vld_dataloader


def get_test_dataloader_and_params(training_input, test_json):
    test_data_loaders = {}
    test_playlists = {}
    r_precisions = {}
    if training_input == 'title':
        for test_seed in test_seeds:
            test_file_path = test_json + '-%d' %test_seed
            with open(test_file_path) as data_file:
                test = json.load(data_file)
                tst_dataset = PlaylistTitleDataset(test['playlists'], track_count, mode='tst')
                test_playlists[test_seed] = test['playlists']
                test_data_loaders[test_seed] =  DataLoader(dataset=tst_dataset, batch_size = BATCH_SIZE, shuffle=False)
                r_precisions[test_seed] = []
    elif training_input == 'track':
        for test_seed in test_seeds:
            test_file_path = test_json + '-%d' %test_seed
            with open(test_file_path) as data_file:
                test = json.load(data_file)
                tst_dataset = PlaylistTrackDataset(test['playlists'], track_count, mode='tst')
                test_playlists[test_seed] = test['playlists']
                test_data_loaders[test_seed] =  DataLoader(dataset=tst_dataset, batch_size = BATCH_SIZE, shuffle=False)
                r_precisions[test_seed] = []
                
    else:
        raise ValueError('Incorrect training input')
                
    return test_data_loaders, test_playlists, r_precisions


def next_title_batch(start_idx, playlists):
    batch_size = min(BATCH_SIZE, len(playlists) - start_idx)
    tensor_in = torch.zeros(batch_size, SEQ_LEN, NUM_CHAR)
    tensor_out = torch.zeros(batch_size, 1, track_count)
    
    for i in range(start_idx, start_idx+batch_size):
        title_indices = playlists[i][2]
        track_indices = playlists[i][0]
        for j in range(SEQ_LEN):
            if title_indices[j] == -1:
                break
            tensor_in[i-start_idx][j][title_indices[j]] = 1
            
        for idx in track_indices:
            tensor_out[i-start_idx][0][idx] = 1
            
    return tensor_in, tensor_out

def record_r_precisions(dataloader, model, playlists, test_seed, record_predictions = False, mode = 'track'):
    num_batches = len(dataloader)
    model.eval()
    loss_tot = 0
    start = 0
    tot_r_prec = 0
    count = 0
    if record_predictions:
        predictions = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = torch.squeeze(pred, dim=1)
            y_pred = pred.detach().cpu().numpy()
            assert y_pred.shape[0] <= BATCH_SIZE
            #print(y_pred.shape)
            for i in range(y_pred.shape[0]):
                inputs, _, title_indices, answers = playlists[start + i]
                #print(y_pred[i, :].shape)
                candidates = np.argsort(-1 * y_pred[i, :]).tolist()[:1000]
                #print(candidates.shape)
                for track_id in inputs:
                    try:
                        candidates.remove(track_id)
                    except:
                        pass
                    
                candidates = candidates[:len(answers)]
                assert len(candidates) == len(answers)
                r_prec = len(set(answers) & set(candidates)) / len(answers)
                tot_r_prec += r_prec
                count += 1
                if mode == 'track' and len(inputs) == 0:
                    tot_r_prec -= r_prec
                    count -= 1
               
                if record_predictions:
                    if mode == 'track':
                        predictions.append((inputs, answers, candidates))
                    elif mode == 'title':
                        cur_title = ''
                        for c_idx in title_indices:
                            if c_idx == -1: 
                                break
                            cur_title += idx_to_char[c_idx]
                        predictions.append((cur_title, answers, candidates))
                            
                
            start += BATCH_SIZE
                
    r_precisions[test_seed].append(tot_r_prec / count)
    if record_predictions:
       with open("%s_pred_test%d" %(mode, test_seed), "wb") as fp:   #Pickling
           pickle.dump(predictions, fp)
       
     


   
           
            
    


    


def record_loss(dataloader, model, loss_fn, loss_vals):
    num_batches = len(dataloader)
    model.eval()
    loss_tot = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_tot += loss_fn(pred, y).item()
    loss_tot /= num_batches
    

    loss_vals.append(loss_tot)

        
    return loss_tot
  





def train(dataloader, model, loss_fn, optimizer, training_loss_vals):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_tot = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_tot += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(loss)
            
        
    loss_tot /= num_batches
    training_loss_vals.append(loss_tot)
    return  loss_tot



def plot_and_save_rprecisions(epochs_list, r_precisions, test_seeds, training_input):
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    for test_seed in test_seeds:
        r_scores = r_precisions[test_seed]
        ax.plot(epochs_list, r_scores, lw = 2, label= r'$N_0 = %d$'  %test_seed)
        
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'Epochs', fontsize=12)
    ax.set_ylabel('R-precision', fontsize=12)
    leg = ax.legend(handlelength=1, fontsize=12)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    plt.tight_layout()
    plt.show()
    fig.savefig('Test-set r precisions %s.pdf' %training_input)
    plt.close(fig)
    
    
    df = pd.DataFrame(data = r_precisions)
    df.to_csv('rprecisions.csv')
    
    
def plot_and_save_losses(epochs_list, training_loss_vals, validation_loss_vals, training_input):
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.plot(epochs_list, training_loss_vals, color='blue', lw = 2, label= 'trn' )
    ax.plot(epochs_list, validation_loss_vals, color='red', lw = 2, label= 'vld' )
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    leg = ax.legend(handlelength=1, fontsize=12)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    plt.tight_layout()
    plt.show()
    fig.savefig('Losses_%s.pdf' %training_input)
    plt.close(fig)
    df = pd.DataFrame(data = {'trn losses': training_loss_vals, 'vld losses': validation_loss_vals})
    df.to_csv('losses.csv')
    




if __name__ == '__main__':
    parser  = argparse.ArgumentParser(description = 'AutoEncoder-based music recommendation system')
    parser.add_argument('--input',               type=str,   default = 'track',  help = 'Input to data encoder.'      )
    parser.add_argument('--trainingjsonfile',    type=str,   default = '../training_data/train',  help = 'JSON file with data for training.' )
    parser.add_argument('--testingjsonfile',     type=str,   default = '../test_data/test',  help = 'JSON files with data for testing.'      )
    parser.add_argument('--validationjsonfile',  type=str,   default = '../valid_data/valid',  help = 'JSON file with data for validation.'    )
    parser.add_argument('--epochs',   type=int, default = 21,   help = 'Number of epochs to train model.' )
    args   = parser.parse_args()
    
    
    
    with open(args.trainingjsonfile) as data_file:
        trn_data = json.load(data_file)
        
    with open(args.validationjsonfile) as data_file:
        vld_data = json.load(data_file)
    
    trn_playlists  = trn_data['playlists']
    char_to_idx    = trn_data['char_to_idx']
    idx_to_char    =  {v: k  for k, v in char_to_idx.items()}
    vld_playlists  = vld_data['playlists']
    track_count    = len(trn_data['track_uri_to_id'])
    SEQ_LEN        = int(trn_data['max_title_len'])
    NUM_CHAR       = int(trn_data['num_char'])
    BATCH_SIZE     = 100
    
    test_seeds = [1, 5, 10, 25, 100]
    training_loss_vals   = []
    validation_loss_vals = []
    
    
    
    
    
    trn_dataloader, vld_dataloader                  = get_trn_vld_dataloaders(args.input, trn_playlists, vld_playlists)
    test_data_loaders, test_playlists, r_precisions = get_test_dataloader_and_params(args.input, args.testingjsonfile)
    
    
    
    model = get_model(args.input)
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = args.epochs
    epochs_list        = np.arange(epochs) + 1
    record_predictions = False
    for t in range(epochs):
        print("Epoch: %d" %(t))
        
        loss_trn = train(trn_dataloader, model, loss_fn, optimizer, training_loss_vals)
        
        #loss_trn = record_loss(trn_dataloader, model, loss_fn, training_loss_vals)
        loss_vld = record_loss(vld_dataloader, model, loss_fn, validation_loss_vals)
        
        if t == epochs-1:
            record_predictions = True
        for test_seed in test_seeds:
            record_r_precisions(test_data_loaders[test_seed], model, test_playlists[test_seed], test_seed, record_predictions=record_predictions, mode=args.input)
            
        print('Loss vals, trn: %g vld: %g' %(loss_trn, loss_vld))
        print(r_precisions)
        
    print("Done training!")
    
    
    plot_and_save_rprecisions(epochs_list, r_precisions, test_seeds, args.input)
    plot_and_save_losses(epochs_list, training_loss_vals, validation_loss_vals, args.input)
    
    torch.save(model.state_dict(), '%s_model.pt' %args.input)