from torch import nn
import torch
import copy
import random
import numpy as np

from torch.utils.data import DataLoader
from dataset_utils import UnsupervisedDataset, SupervisedDataset

import time
from tqdm import tqdm

from torch.optim import Adam, RMSprop, SGD
from utils import LRScheduler, EarlyStopping


def client_train_unsupervised(args, model, data_train, data_valid):
    
    model.to(args.device)

    dataset_train = UnsupervisedDataset(data_train)
    dataset_valid = UnsupervisedDataset(data_valid)

    train_generator = DataLoader(dataset_train, 
                                 batch_size=args.B, 
                                 shuffle=True,
                                 drop_last=True, 
                                 num_workers=0)
    
    valid_generator = DataLoader(dataset_valid, 
                                 batch_size=args.B, 
                                 shuffle=True, 
                                 drop_last=True,
                                 num_workers=0)


    # Training parameters
    epochs = 1000
    min_valid_loss = np.inf
    # lr = 0.001

    criterion = NTXentLoss(args)

    optimizer = Adam([{'params': model.parameters(), 'lr': 3e-4}])
    # optimizer = RMSprop([{'params': model.parameters(), 'lr': 0.001}, {'params': metric_fc.parameters(), 'lr': 0.001}])

    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    training_loss= []
    valid_loss= []

    for epoch in range(epochs):  # loop over the dataset multiple times

        model.train()
        training_running_loss = 0.0
        training_running_correct = 0

        # i = 0
        for iteration, (data_left, data_right) in enumerate(tqdm(train_generator)):
            
            data_left = data_left.to(args.device)
            data_right = data_right.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            feature_left = model(data_left)
            feature_right = model(data_right)

            loss = criterion(feature_left, feature_right)

            loss.backward()
            optimizer.step()

            # print statistics
            training_running_loss += loss.item()

        training_epoch_loss = training_running_loss / (iteration + 1)

        model.eval()
        valid_running_loss = 0.0

        # i = 0
        for iteration, (data_left, data_right) in enumerate(valid_generator):

            data_left = data_left.to(args.device)
            data_right = data_right.to(args.device)

            feature_left_v = model(data_left)
            feature_right_v = model(data_right)

            loss = criterion(feature_left_v, feature_right_v)

            # print statistics
            valid_running_loss += loss.item()

        valid_epoch_loss = valid_running_loss / (iteration + 1)


        print('Epoch '+str(epoch+1)
              +'\t Training Loss: '+str(round(training_epoch_loss, 2))
              +'\t Validation Loss: '+str(round(valid_epoch_loss, 2))
              +'\t')

        training_loss.append(training_epoch_loss)
        valid_loss.append(valid_epoch_loss)


        best_model = copy.deepcopy(model)
        
        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            break

    print('Finished Training')

    return best_model




def client_train_supervised(args, model, data_train, data_valid, label_train, label_valid, FineTune=False):

    model.to(args.device)

    num_class = label_train.shape[1]

    metric_fc = nn.Sequential(
                nn.Linear(128, 64),  
                nn.ReLU(),
                nn.Linear(64, num_class)
                            )
    # metric_fc = nn.Linear(128, num_class)

    metric_fc.to(args.device)


    dataset_train = SupervisedDataset(data_train, label_train)
    dataset_valid = SupervisedDataset(data_valid, label_valid)

    train_generator = DataLoader(dataset_train,
                                batch_size=args.B,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True)
    
    valid_generator = DataLoader(dataset_valid,
                                batch_size=args.B,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True )

    # Training parameters
    epochs = 1000
    min_valid_loss = np.inf
    # lr = 0.001

    criterion = nn.CrossEntropyLoss()

    if FineTune:
        optimizer = Adam([{'params': model.parameters(), 'lr': 1e-3}, 
                          {'params': metric_fc.parameters(), 'lr': 1e-3}])
    else:
        optimizer = Adam([{'params': model.parameters(), 'lr': 3e-4}, 
                          {'params': metric_fc.parameters(), 'lr': 3e-4}])
      
    # optimizer = RMSprop([{'params': model.parameters(), 'lr': 0.001}, {'params': metric_fc.parameters(), 'lr': 0.001}])

    lr_scheduler = LRScheduler(optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    early_stopping = EarlyStopping()

    training_loss, training_acc = [], []
    valid_loss, valid_acc = [], []


    for epoch in range(epochs):  # loop over the dataset multiple times

        model.train()
        training_running_loss = 0.0
        training_running_correct = 0

        for iteration, (inputs, labels) in enumerate(train_generator):

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            feature = model(inputs)
            outputs = metric_fc(feature)
            # outputs = metric_fc(feature, labels)

            preds = torch.argmax(outputs, dim=1)
            truth = torch.argmax(labels, dim=1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            training_running_loss += loss.item()
            training_running_correct += (preds == truth).sum().item()


        training_epoch_loss = training_running_loss / (iteration + 1)
        training_epoch_acc = 100.0 * training_running_correct / ((iteration + 1) * args.B)

        model.eval()
        valid_running_loss = 0.0
        valid_running_correct = 0

        for iteration, (inputs, labels) in enumerate(valid_generator):

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            feature_v = model(inputs)
            outputs_v = metric_fc(feature_v)
            # outputs_v = metric_fc(feature_v, labels)

            loss = criterion(outputs_v, labels)

            # preds = torch.argmax(outputs_v, dim=1)
            # truth = torch.argmax(labels, dim=1)

            # print statistics
            valid_running_loss += loss.item()
            valid_running_correct += (preds == truth).sum().item()

        valid_epoch_loss = valid_running_loss / (iteration + 1)
        valid_epoch_acc = 100.0 * valid_running_correct / ((iteration + 1) * args.B)

        # print('Epoch ' + str(epoch + 1) + '\t Training Loss: ' + str(
        #     training_epoch_loss) + '\t Validation Loss: ' + str(valid_epoch_loss))

        print('Epoch '+str(epoch+1)
              +'\t Training Loss: '+str(round(training_epoch_loss, 2))
              +'\t Validation Loss: '+str(round(valid_epoch_loss, 2))
              +'\t Training Acc: '+str(round(training_epoch_acc,2))
              +'\t Validation Acc: '+str(round(valid_epoch_acc, 2))
            #   +'\t LR: '+str(round(lr_scheduler.get_last_lr()[0], 4))
              +'\t')

        training_loss.append(training_epoch_loss)
        training_acc.append(training_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)


        best_model = copy.deepcopy(model)
        best_clf = copy.deepcopy(metric_fc)
        
        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            break

    print('Finished Training')

    return best_model, best_clf
    # return model, metric_fc




class NTXentLoss(torch.nn.Module):

    def __init__(self, args, temperature_or_m=0.05, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = args.B
        self.temperature = temperature_or_m
        self.device = args.device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        # print(loss)
        # o1 = torch.nn.LogSoftmax(dim=1)
        # o2 = torch.nn.NLLLoss(reduction='none')
        # p1 = o1(logits)
        # print(p1)
        # p2 = o2(p1, labels)
        # print(p2)

        return loss / (2 * self.batch_size)