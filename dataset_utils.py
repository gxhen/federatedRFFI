import argparse
import torch
import numpy as np
import pandas as pd
import random
import h5py
from scipy import signal
import math 

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from pyphysim.channels.fading import COST259_TUx, COST259_RAx, TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator, RayleighSampleGenerator


class UnsupervisedDataset(Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]

        data_left = self._process_data(data_input)
        data_right = self._process_data(data_input)

        return data_left, data_right

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        data_out = channel_ind_spectrogram(data_out[0], win_len=64, crop_ratio=0.3)
        data_out = torch.from_numpy(data_out.astype(np.float32))
        return data_out


class SupervisedDataset(Dataset):
    def __init__(self, data_input, label_input):
        self.data_input = data_input
        self.label_input = label_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]

        data_input= self._process_data(data_input)
        label_input = label_input.astype(float)
        return data_input, label_input

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        data_out = channel_ind_spectrogram(data_out[0], win_len=64, crop_ratio=0.3)
        # view_spec(data_out[0])
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out


def view_spec(input_spec):
    plt.figure()
    sns.heatmap(input_spec, cmap='crest', cbar=False,
                xticklabels=False, yticklabels=False, ).invert_yaxis()
    plt.savefig('spectrogram.pdf', bbox_inches='tight')


def awgn(data, snr_range):
    if len(data.shape) == 1:
        data = data.reshape(1, len(data))
        
    data_noisy = np.zeros(data.shape, dtype=complex)
    pkt_num = data.shape[0]
    SNRdB = np.random.uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(len(s)) + 1j * np.random.standard_normal(len(s)))
        data_noisy[pktIdx] = s + n

    return data_noisy

def cal_exponential_pdp(tau_d, Ts, A_dB = -30):

    # Exponential PDP generator
    # Inputs:
    # tau_d : rms delay spread[sec]
    # Ts : Sampling time[sec]
    # A_dB : smallest noticeable power[dB]
    # norm_flag : normalize total power to unit
    # Output:
    # PDP : PDP vector

    sigma_tau = tau_d
    A = 10**(A_dB / 10)
    lmax = np.ceil(-tau_d * np.log(A) / Ts)
    
    # Exponential PDP
    p = np.arange(0, lmax+1)
    pathDelays = p * Ts
    
    p = (1 / sigma_tau) * np.exp(-p * Ts / sigma_tau)
    p_norm = p / np.sum(p)
    
    avgPathGains = 10 * np.log10(p_norm)
    
    return avgPathGains, pathDelays


def data_aug_operator(data_in, multipath=True):

    if multipath:
        # data_out = np.zeros(data_in.shape, dtype=complex)
        Ts = 1/500000
            
        tau_d = np.random.uniform(5, 300)*1e-9
        Fd = np.random.uniform(0, 5)
        # Create a jakes object with 20 rays. This is the fading model that controls how the channel vary in time.
        # This will be passed to the TDL channel object.
        chObj = JakesSampleGenerator(Fd=Fd, Ts=Ts, L=5)
        # chObj = RayleighSampleGenerator()
        avgPathGains, pathDelays = cal_exponential_pdp(tau_d, Ts)

        # Creates the tapped delay line (TDL) channel model, which accounts for the multipath and thus the
        # frequency selectivity
        pdpObj = TdlChannelProfile(avgPathGains,
                                   pathDelays,
                                   'Exponential_PDP')

        tdlchannel = TdlChannel(chObj, pdpObj)
        
        data_corrputed = tdlchannel.corrupt_data(data_in)
        data_out = data_corrputed[:len(data_corrputed)-tdlchannel.num_taps+1]
            # cir = tdlchannel.get_last_impulse_response()

        data_out = awgn(data_out, snr_range = range(80))

    else:
        data_out = awgn(data_in, snr_range = range(80))

    return data_out


def normalization(data):
    ''' Normalize the signal.'''
    
    amplitude = np.abs(data)
    rms = np.sqrt(np.mean(amplitude**2))
    data_norm = data/rms
    
    return data_norm


def channel_ind_spectrogram(data, win_len = 64, crop_ratio = 0.3):
    ''' Generate channel independent spectrogram.'''
    def _spec_crop(x, crop_ratio):
    
        num_row = x.shape[0]
        x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

        return x_cropped

    f, t, spec = signal.stft(data, 
                            window='boxcar', 
                            nperseg= win_len, 
                            noverlap= round(0.5*win_len), 
                            nfft= win_len,
                            return_onesided=False, 
                            padded = False, 
                            boundary = None)
    
    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    # spec = spec_crop(spec, crop_ratio)
    spec = spec + 1e-12
    
    data_out = spec[:,1:]/spec[:,:-1]    
                
    data_out = np.log10(np.abs(data_out)**2)

    data_out = _spec_crop(data_out, crop_ratio)
    
    data_out = np.expand_dims(data_out, axis=0)

    return data_out

# class ChannelIndSpectrogram():
#     def __init__(self,):
#         pass
    
#     def _normalization(self,data):
#         ''' Normalize the signal.'''
#         s_norm = np.zeros(data.shape, dtype=complex)
        
#         for i in range(data.shape[0]):
        
#             sig_amplitude = np.abs(data[i])
#             rms = np.sqrt(np.mean(sig_amplitude**2))
#             s_norm[i] = data[i]/rms
        
#         return s_norm        

#     def _spec_crop(self, x, crop_ratio):
      
#         num_row = x.shape[0]
#         x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

#         return x_cropped

#     def _gen_single_channel_ind_spectrogram(self, sig, win_len, crop_ratio=0.3):
        
#         sig = self._normalization(sig)
#         overlap = round(0.5*win_len)

#         f, t, spec = signal.stft(sig[0], 
#                                 window='boxcar', 
#                                 nperseg= win_len, 
#                                 noverlap= overlap, 
#                                 nfft= win_len,
#                                 return_onesided=False, 
#                                 padded = False, 
#                                 boundary = None)
        
#         # spec = spec_shift(spec)
#         spec = np.fft.fftshift(spec, axes=0)
#         # spec = spec_crop(spec, crop_ratio)
#         spec = spec + 1e-12
        
#         dspec = spec[:,1:]/spec[:,:-1]    
                 
#         dspec_amp = np.log10(np.abs(dspec)**2)
#         # dspec_phase = np.angle(dspec)
#         dspec_amp = self._spec_crop(dspec_amp, crop_ratio)
        
#         dspec_amp = np.expand_dims(dspec_amp, axis=0)

#         return dspec_amp
    

    # def channel_ind_spectrogram(self, data, win_len = 64, crop_ratio = 0):
    #     data = self._normalization(data)
        
    #     # win_len = 16
    #     overlap = 0.5
        
    #     num_sample = data.shape[0]
    #     # num_row = math.ceil(win_len*(1-2*crop_ratio))
    #     num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    #     num_column = int(np.floor((data.shape[1]-win_len)/(win_len - round(overlap*win_len))) + 1) - 1
        
        
    #     data_dspec = np.zeros([num_sample, 1, num_row, num_column,])
    #     # data_dspec = []
    #     for i in range(num_sample):
                   
    #         dspec_amp = self._gen_single_channel_ind_spectrogram(data[i], win_len, round(overlap*win_len))
    #         dspec_amp = self._spec_crop(dspec_amp, crop_ratio)
    #         data_dspec[i,0,:,:] = dspec_amp
    #         # data_dspec[i,:,:,1] = dspec_phase
            
    #     return data_dspec   

    # def multi_resolution_spec(self, data, args, crop_ratio = 0.3):
    #     data = self._normalization(data)
    #     num_sample = data.shape[0]

    #     # win_len = 16
    #     overlap = 0.5
    #     # win_len_group = [64, 128, 256]
        
    #     data_dspec_group = []
    #     for win_len in args.wingroup:
            
    #         # num_row = math.ceil(win_len*(1-2*crop_ratio))
    #         num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    #         num_column = int(np.floor((data.shape[1]-win_len)/(win_len - round(overlap*win_len))) + 1) - 1
            
            
    #         data_dspec = np.zeros([num_sample, 1, num_row, num_column,])
    #         # data_dspec = []
    #         for i in range(num_sample):
                    
    #             dspec_amp = self._gen_single_channel_ind_spectrogram(data[i], win_len, round(overlap*win_len))
    #             dspec_amp = self._spec_crop(dspec_amp, crop_ratio)
    #             data_dspec[i,0,:,:] = dspec_amp
    #             # data_dspec[i,:,:,1] = dspec_phase

    #         data_dspec_group.append(data_dspec)

    #     return data_dspec_group

    # def view_spec(self, input_spec):
    #     # plt.figure()
    #     # plt.imshow(input_spec, cmap='jet', origin='lower')
    #     # plt.show(block=True)

    #     # cmap 'Blues' default
    #     plt.figure()
    #     sns.heatmap(input_spec[0, 0, :, :], cmap='crest', cbar=False,
    #                 xticklabels=False, yticklabels=False, ).invert_yaxis()
    #     plt.savefig('spectrogram.pdf', bbox_inches='tight')