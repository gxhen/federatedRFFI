import argparse
import torch
import numpy as np
import pandas as pd
import random
from torch.autograd import Variable
import h5py
from scipy import signal
import math 

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from pyphysim.channels.fading import COST259_TUx, COST259_RAx, TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator, RayleighSampleGenerator


class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        
    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1] 
        data_complex = np.zeros([num_row,round(num_col/2)],dtype=complex)
     
        data_complex = data[:,:round(num_col/2)] + 1j*data[:,round(num_col/2):] 
        return data_complex

    
    def load_iq_samples(self, file_path, downsampling=True):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        data = []
        label = []

        for file_name in file_path:

            f = h5py.File(file_name,'r')
            label_temp = f[self.labelset_name][:]
            label_temp = np.transpose(label_temp)

            data_temp = f[self.dataset_name]
            data_temp = self._convert_to_complex(data_temp)

            if downsampling:
                sig_len = data_temp.shape[1]
                data_temp = data_temp[:, 0:sig_len:2]

            f.close()

            data.extend(data_temp)
            label.extend(label_temp)

        data = np.array(data)
        label = np.array(label)

        return data, label
    
    def load_iq_samples_range(self, file_path, pkt_range, downsampling=True):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        
        f = h5py.File(file_path,'r')
        label = f[self.labelset_name][:]
        label = np.transpose(label)

        sample_index_list = []
        
        for dev_idx in np.unique(label):
            sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
    
        data = f[self.dataset_name][:]
        data = data[sample_index_list]
        data = self._convert_to_complex(data)
        
        if downsampling:
            sig_len = data.shape[1]
            data = data[:, 0:sig_len:2]

        label = label[sample_index_list]
        
        f.close()
        return data, label
    
    def load_multiple_files(self, file_list, tx_range, pkt_range):

        # file_list = os.listdir(folder_name)
        # file_list = natsorted(file_list, key=lambda y: y.lower())
        
        # file_list = [file_list[i] for i in rx_range]
        
        num_rx = len(file_list)
        num_tx = len(tx_range)
        num_pkt = len(pkt_range)
        
        data = []
        tx_label = []
        rx_label = []
        
        for file_idx in range(num_rx):
            print('Start loading dataset ' + str(file_idx + 1))
            filename = file_list[file_idx]
            # filename = folder_name + filename
            [data_temp, tx_label_temp, _ ] = self.load_iq_samples(filename, tx_range, pkt_range)
            rx_label_temp = np.ones(num_pkt*num_tx)*file_idx
            
            data.extend(data_temp)
            tx_label.extend(tx_label_temp)
            rx_label.extend(rx_label_temp)
            
        data = np.array(data)   
        tx_label = np.array(tx_label)
        rx_label = np.array(rx_label)
        
        return data, tx_label, rx_label


class LRScheduler:

    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:

    def __init__(self, patience=20, min_delta=0):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

    

def plt_cm(true_label, pred_label, num_predictable_classes):
    conf_mat = confusion_matrix(true_label, pred_label)
    classes = range(1, num_predictable_classes+1)

    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.show(block=True)


def to_onehot(label_in):

        u, label_int = np.unique(label_in, return_inverse=True)
        num_classes = len(u)
        label_one_hot = np.eye(num_classes, dtype='uint8')[label_int]

        return label_one_hot, num_classes