import torch
import numpy as np
import argparse
from models import SimCLRNet
from utils import  plt_cm, to_onehot, LoadDataset
import matplotlib.pyplot as plt
import time

from dataset_utils import channel_ind_spectrogram, awgn

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from client import client_train_supervised

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--B', type=int, default=16, help='Fine tune batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args



class Test:
    def __init__(self, args, data_finetune_in, label_finetune_in, cnn_name_in):

        self.args = args

        label_finetune, self.num_classes = to_onehot(label_finetune_in)
    

        self.data_train, self.data_valid, self.label_train, self.label_valid = train_test_split(data_finetune_in,
                                                                                                label_finetune,
                                                                                                test_size=0.1,
                                                                                                shuffle=True)


        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", self.device, "device")

        self.model = SimCLRNet(name='server')
        
        if cnn_name_in is not None:
            self.model.load_state_dict(torch.load(cnn_name_in))
            print('Pre-trained model has been loaded.')
        else:
            print('No pre-trained model is loaded.')

        self.model.to(self.device)
        self.model.eval()

    def fine_tune(self):
        
        print('Start fine-tune.')

        self.model_tuned, self.clf_tuned = client_train_supervised(self.args, 
                                                                    self.model, 
                                                                    self.data_train, 
                                                                    self.data_valid, 
                                                                    self.label_train, 
                                                                    self.label_valid,
                                                                    FineTune=True)
        

        print('Fine-tune complete.')


    def test(self, data_test_in, label_test_in, snr=None, plot_confusion_matrix=True):

        label_test, self.num_classes = to_onehot(label_test_in)
        self.label_test = label_test.argmax(axis=-1)
        self.data_test = data_test_in

        if len(self.data_test.shape) == 1:
            self.data_test = self.data_test.reshape([1, len(self.data_test)])

        if snr is not None:
            data_test_temp = awgn(self.data_test, [snr])
        else:
            data_test_temp = self.data_test

        pred_label = []
        for i in range(0, data_test_temp.shape[0]):
            test_input = data_test_temp[i:i + 1]


            test_input = channel_ind_spectrogram(test_input[0], win_len=64, crop_ratio=0.3)
            test_input = np.expand_dims(test_input, axis=0)

            test_input = torch.from_numpy(test_input.astype(np.float32))
            test_input = test_input.to(self.device)  # Transfer to GPU


            start = time.time()
            feature = self.model_tuned(test_input)
            test_output = self.clf_tuned(feature)
            end = time.time()
            inference_time = end - start

            test_output = test_output.cpu()
            test_output = test_output.detach().numpy()

            # Used for test-time augmentation
            # test_output = np.mean(test_output, axis=0)

            test_output = test_output.argmax(axis=-1)
            pred_label.extend(test_output)

        pred_label = np.array(pred_label)
        # print('Inference done. ' + str(len(feature_out)) + ' predictions.')

        acc = accuracy_score(self.label_test, pred_label)
        print('Accuracy on query set is ' + str(acc))
        if plot_confusion_matrix:
            plt_cm(self.label_test, pred_label, self.num_classes)

        return acc



if __name__ == '__main__':
    args = args_parser()

    nn_server_name = './model/nn_fed_comm_round_5.pth'


    LoadDatasetObj = LoadDataset()

    file_path_finetune = './dataset/test/N210_A_test.h5'

    data_finetune, label_finetune = LoadDatasetObj.load_iq_samples_range(file_path_finetune,
                                                                         pkt_range=range(0, 200))


    TestObj = Test(args, data_finetune, label_finetune, nn_server_name)

    TestObj.fine_tune()


    file_path_test = './dataset/test/N210_A_test.h5'

    data_test, label_test = LoadDatasetObj.load_iq_samples_range(file_path_test,
                                                                 pkt_range=range(200, 500))

    acc = TestObj.test(data_test, label_test, plot_confusion_matrix=True)


    ## Test the model with different SNR

    # acc_snr = []
    # for snr_add in range(10, 50, 5):
    #     acc = TestObj.test(data_test, label_test, snr=snr_add, plot_confusion_matrix=False)
    #     acc_snr.append(acc)
    # print(acc_snr)


    
