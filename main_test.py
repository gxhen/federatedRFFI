import torch
import numpy as np
import argparse
from models import SimCLRNet
from utils import  plt_cm, to_onehot, LoadDataset
import matplotlib.pyplot as plt
import time

from dataset_utils import channel_ind_spectrogram, awgn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split


from client import client_train_supervised

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--B', type=int, default=16, help='Fine tune batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args


class DistMeasure:
    def __init__(self, args, data_support_in, label_support_in, data_query_in, label_query_in, cnn_name_in):

        self.args = args

        self.label_support, self.num_classes = to_onehot(label_support_in)
        self.label_query, self.num_classes = to_onehot(label_query_in)

        self.label_support = self.label_support.argmax(axis=-1)
        self.label_query = self.label_query.argmax(axis=-1)

        self.data_support = data_support_in
        self.data_query = data_query_in

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", self.device, "device")

        # self.model = SimpleCNN(name='server')
        self.model = SimCLRNet(name='server')
        self.model.load_state_dict(torch.load(cnn_name_in))
        self.model.to(self.device)
        self.model.eval()

    def _extract_feature(self, data_test):

        if len(data_test.shape) == 1:
            data_test = data_test.reshape([1, len(data_test)])

        model_out = []
        for i in range(0, data_test.shape[0]):
            test_input = data_test[i:i + 1]
            
            test_input = channel_ind_spectrogram(test_input[0], win_len=64, crop_ratio=0.3)
            test_input = np.expand_dims(test_input, axis=0)

            test_input = torch.from_numpy(test_input.astype(np.float32))
            test_input = test_input.to(self.device)  # Transfer to GPU


            start = time.time()
            test_output = self.model(test_input)
            end = time.time()
            inference_time = end - start

            test_output = test_output.cpu()
            test_output = test_output.detach().numpy()

            # Used for test-time augmentation
            # test_output = np.mean(test_output, axis=0)

            model_out.extend(test_output)

        model_out = np.array(model_out)
        # print('Inference done. ' + str(len(feature_out)) + ' predictions.')

        return model_out

    def build_database(self, augmentation=True):  
        print('Start building RFF database.')  
    
        if augmentation:  
            # augment the support set 10 times  
            self.data_support = np.repeat(self.data_support, 10, axis=0)  
            self.label_support = np.repeat(self.label_support, 10, axis=0)  
    
            self.data_support = awgn(self.data_support, range(50))  
    
        feature_enrol = self._extract_feature(self.data_support)  
    
        # Compute the mean vector for each class  
        self.database = {}  
        for label in np.unique(self.label_support):  
            idx = np.where(self.label_support == label)[0]  
            self.database[label] = np.mean(feature_enrol[idx], axis=0)  
    
        print('RFF database is built')  
    
    def dist_measure_test(self, snr, plot_confusion_matrix=False):  
        def _compute_cosine_similarity(x, y):
            res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            return res

        if snr is not None:  
            data_noisy = awgn(self.data_query, [snr])  
            feature_query = self._extract_feature(data_noisy)  
        else:  
            feature_query = self._extract_feature(self.data_query)  

        pred_prob = np.zeros([len(feature_query), self.num_classes])
        # Compute cosine similarity between the query feature and each class mean vector  
        for pkt_ind in range(len(feature_query)):
            for label_ind in range(self.num_classes):  
                pred_prob[pkt_ind, label_ind] = _compute_cosine_similarity(feature_query[pkt_ind], self.database[label_ind])
    
        # Find the label with the highest similarity and assign it as the predicted label  
        pred_label = pred_prob.argmax(axis=-1)
    
        acc = accuracy_score(self.label_query, pred_label)  
        print('Accuracy on query set is ' + str(acc))  
        if plot_confusion_matrix:  
            plt_cm(self.label_query, pred_label, self.num_classes)  
    
        return acc

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

    # nn_server_name = './model_baseline/nn_central.pth'
    nn_server_name = './model_baseline/nn_fed_comm_round_5.pth'
    


    LoadDatasetObj = LoadDataset()

    file_path_finetune = './fed_rffi_dataset/N210_A_test.h5'

    data_finetune, label_finetune = LoadDatasetObj.load_iq_samples_range(file_path_finetune,
                                                                         pkt_range=range(0, 200))


    TestObj = Test(args, data_finetune, label_finetune, nn_server_name)

    TestObj.fine_tune()

    file_path_test = './fed_rffi_dataset/N210_A_test.h5'

    data_test, label_test = LoadDatasetObj.load_iq_samples_range(file_path_test,
                                                                 pkt_range=range(200, 500))

    # acc = TestObj.test(data_test, label_test, plot_confusion_matrix=True)

    acc_snr = []
    for snr_add in range(10, 50, 5):
        acc = TestObj.test(data_test, label_test, snr=snr_add, plot_confusion_matrix=False)
        acc_snr.append(acc)
    print(acc_snr)


    
