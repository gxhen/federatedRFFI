from models import SimCLRNet
import copy
import torch
from utils import LoadDataset, to_onehot
from client import client_train_unsupervised, client_train_supervised

from sklearn.model_selection import train_test_split
import argparse
import time

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default='federated', help='federated or centralized training')
    parser.add_argument('--r', type=int, default=5, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=4, help='number of total clients')
    parser.add_argument('--B', type=int, default=128, help='local batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args


class FedTrain:
    def __init__(self, args):
        self.args = args
        self.nn_server = SimCLRNet(name='server').to(args.device)
        print("The model will be running on", args.device, "device")
        self.nn_clients = []
        self.clf_clients = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn_server)
            # clients = ['AP_' + str(i) for i in range(1, 4)]
            # temp.name = self.args.clients[i]
            temp.name = 'client_' + str(i+1)
            self.nn_clients.append(temp)

    def server(self):
        
        for round_ind in range(self.args.r):
            print('Federated training round', round_ind + 1, ':')
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # dispatch
            self.dispatch()
            # local updatingll
            self.client_update(round_ind)
            # aggregation
            self.aggregation()
            torch.save(self.nn_server.state_dict(), './model/nn_fed_comm_round_' + str(round_ind + 1) + '.pth')

        print('Federated training done.')
        return self.nn_server

    def aggregation(self):
        norm_factor = 0
        for j in range(self.args.K):
            # normal
            norm_factor += self.nn_clients[j].dataset_size

        # zero the parameters of the server nn
        for v in self.nn_server.parameters():
            v.data.zero_()

        for j in range(self.args.K):
            # cnt = 0
            for v1, v2 in zip(self.nn_server.parameters(), self.nn_clients[j].parameters()):
                v1.data += v2.data * (self.nn_clients[j].dataset_size / norm_factor)
                # cnt += 1
                # if cnt == 2 * (self.args.total - self.args.Kp):
                #     break

    def dispatch(self):
        for j in range(self.args.K):
            # cnt = 0
            for old_params, new_params in zip(self.nn_clients[j].parameters(), self.nn_server.parameters()):
                old_params.data = new_params.data.clone()
                # cnt += 1
                # if cnt == 2 * (self.args.total - self.args.Kp):
                #     break

    def client_update(self, round_ind):  # update nn

        for k in range(self.args.K):
            model_client = self.nn_clients[k]
            dataset_path = ['D:/fed_rffi_dataset/' + model_client.name + '_train.h5']

            LoadDatasetObj = LoadDataset()

            data, label = LoadDatasetObj.load_iq_samples(dataset_path)

            print('Load training data from ' + model_client.name + '...')

            label_one_hot, num_class = to_onehot(label)
            print('Training data is collected from ' + str(num_class) + ' devices.')
            data_train, data_valid, label_train, label_valid = train_test_split(data,
                                                                                label_one_hot,
                                                                                test_size=0.1,
                                                                                shuffle=True)
            print('Loading done.')

            model_client.dataset_size = len(data_train) # record the dataset size for normalization in the aggreation process

            # if round_ind == 0:
            #     clf_client = nn.Linear(128, num_class)
            #     self.clf_clients.append(clf_client)
            # else:
            #     clf_client = self.clf_clients[k]
            method = 'unsupervised'

            if method == 'unsupervised':
                print('Start unsupervised training.')
                self.nn_clients[k] = client_train_unsupervised(self.args, 
                                                               model_client,
                                                               data_train, 
                                                               data_valid)
            elif method == 'supervised':
                print('Start supervised training.')
                self.nn_clients[k], _ = client_train_supervised(self.args, 
                                                                model_client, 
                                                                data_train, 
                                                                data_valid, 
                                                                label_train, 
                                                                label_valid)



class CenTrain:
    def __init__(self, args):
        self.args = args
        self.nn_client = SimCLRNet(name='server').to(args.device)
        print("The model will be running on", args.device, "device")

    def start_train(self, dataset_path):

        LoadDatasetObj = LoadDataset()

        data, label = LoadDatasetObj.load_iq_samples(dataset_path)

        label_one_hot, num_class = to_onehot(label)
        data_train, data_valid, label_train, label_valid = train_test_split(data,
                                                                            label_one_hot,
                                                                            test_size=0.1,
                                                                            shuffle=True)
        print('Loading done.')
        

        print('Start centralized training.')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        method = 'unsupervised'

        if method == 'unsupervised':
            self.nn_client = client_train_unsupervised(self.args, 
                                                       self.nn_client,
                                                       data_train, 
                                                       data_valid)
        elif method == 'supervised':
            self.nn_client, _ = client_train_supervised(self.args, 
                                                        self.nn_client, 
                                                        data_train, 
                                                        data_valid, 
                                                        label_train, 
                                                        label_valid)


        print('Training done.')
        return self.nn_client
        


if __name__ == '__main__':

    args = args_parser()

    if args.mode == 'federated':
        fedTrain = FedTrain(args)
        nn_server = fedTrain.server()

    elif args.mode == 'centralized':
        CenTrain = CenTrain(args)
        dataset_path = [
                        './dataset/train/client_1_train.h5',
                        './dataset/train/client_2_train.h5',
                        './dataset/train/client_3_train.h5',
                        './dataset/train/client_4_train.h5'
                        ]
        nn_cen_train = CenTrain.start_train(dataset_path)
        torch.save(nn_cen_train.state_dict(), './model/nn_central.pth')
