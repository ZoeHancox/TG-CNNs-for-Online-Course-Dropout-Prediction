#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from scipy import sparse
from scipy.linalg import sqrtm 

import tensorflow as tf
import tensorflow.keras as keras

from torch.utils.data import TensorDataset, DataLoader
import torch, torch.nn as nn
import torchvision
import torchvision.models as models

import warnings
warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

actions = pd.read_csv('data/raw_data/mooc_actions.tsv', sep='\t')
features = pd.read_csv('data/raw_data/mooc_action_features.tsv', sep='\t')
grp = features.groupby(['FEATURE0', 'FEATURE1', 'FEATURE2', 'FEATURE3'])
labels = pd.read_csv('data/raw_data/mooc_action_labels.tsv', sep='\t')
actions['LABEL'] = labels.LABEL

y = actions.groupby('USERID').LABEL.sum().values

X = np.zeros([len(actions.USERID.unique()), 100]) - 1 # returns a numpy matrix filled with -1's
# shape = length of unique users x 100
# shape = 7047 x 100
for curuser in actions.USERID.unique():
    curactions = actions[actions.USERID == curuser].TARGETID
    if len(curactions) > 100:
        X[curuser, :] = curactions[-100:]
    else:
        X[curuser, -len(curactions):] = curactions
        
X = X[:, :, np.newaxis] # np.newaxis increases the 2D vector by one dim

X = X[:-7]
y = y[:-7]


batch_size = 128


# In[2]:


# Splitting the data into train, val and test sets
x_valtrain, x_test, y_valtrain, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
x_train, x_val, y_train, y_val = train_test_split(x_valtrain, y_valtrain, test_size=0.1, random_state=20)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_val = torch.Tensor(x_val)
y_val = torch.Tensor(y_val)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

train = TensorDataset(x_train, y_train)
val = TensorDataset(x_val, y_val)
test = TensorDataset(x_test, y_test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


# In[3]:


class RNNModel(nn.Module):
    def __init__(self, hidden_dim, fc1_size, fc2_size):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.sig= nn.Sigmoid()

        # RNN layers
        self.rnn = nn.RNN(1, hidden_dim, 1, batch_first=True)
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# In[4]:


def val_test(loader=val_loader):
    val_losses, val_AUCs, val_accs, val_precs, val_recalls, val_f1s = ([] for i in range(6))
    model.eval() # might be needed as some layers e.g. batchnorm perform differently

    with torch.no_grad():
        for inputs, labels in loader:
      
            val_inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            val_labels = labels.float()

            val_out = model(val_inputs.float())
            val_loss = criterion(val_out, val_labels)

            val_losses.append(val_loss.item()) # list all val_losses over the batch
            sig = nn.Sigmoid()
            pred = sig(val_out).round()
            pred = torch.transpose(pred, 0, 1).squeeze()

            labels_detached = labels.cpu().detach().numpy()
            pred_detached = pred.cpu().detach().numpy()
            #print(f"Predicted: {pred.squeeze()}\nActual: {labels.squeeze()}")
            val_AUCs.append(roc_auc_score(labels_detached, pred_detached)) 
            val_precs.append(precision_score(labels_detached, pred_detached))
            val_recalls.append(recall_score(labels_detached, pred_detached))
            val_f1s.append(f1_score(labels_detached, pred_detached))
            val_accs.append(accuracy_score(labels_detached, pred_detached))

    return np.mean(val_losses), np.mean(val_AUCs), np.mean(val_accs), np.mean(val_precs), np.mean(val_recalls), np.mean(val_f1s)


# In[ ]:


nepochs_choice =[5, 10, 15, 20]
lr_choice = [0.1, 0.01, 0.001]
layer1_choice=[16,32,64,128]
layer2_choice=[16,32,64,128]
layer3_choice=[16,32,64]

print("Num of parameter combinations =", len(nepochs_choice)*len(lr_choice)*len(layer1_choice)*len(layer2_choice)*len(layer3_choice))

# nepochs_choice =[5, 10]
# lr_choice = [0.1]
# layer1_choice=[16, 32]
# layer2_choice=[16]
# layer3_choice=[16]

best_rnn = 0
for nepochs in nepochs_choice:
    for lr in lr_choice:
        for layer1 in layer1_choice:
            for layer2 in layer2_choice:
                for layer3 in layer3_choice:
                    print(f"\nParams: epochs={nepochs}, lr={lr}, layer1={layer1}, layer2={layer2}, layer3={layer3}")
    
                    model = RNNModel(hidden_dim = layer1, fc1_size=layer2, fc2_size=layer3)
                    model.to(device)

                    criterion = nn.BCEWithLogitsLoss().to(device) # with logits loss does sigmoid
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    for epoch in range(nepochs):
                        trn_losses, trn_AUCs, trn_accs, trn_precs, trn_recalls, trn_f1s = ([] for i in range(6))
                        acc_TESTER = []
                        for i, batch in enumerate(train_loader):
                            x, labels = batch
                            labels=labels.unsqueeze(1)
                            #print(x)
                            #print(f"\n\n{labels}")
                            inputs = x.to(device)
                            labels = labels.to(device) # this is for accuracy calc
                            targets = labels # to get the same dims as input

                            model.train()
                            outputs = model(inputs.float())
                            train_loss = criterion(outputs, targets) # loss function
                            

                            train_loss.backward()
                            optimizer.step() # update the weights
                            optimizer.zero_grad() # zero parameter gradients to prevent carrying forward values from previous iter

                            sig = nn.Sigmoid()
                            pred = sig(outputs).round()
                            pred = torch.transpose(pred, 0, 1).squeeze()
                            #print(pred)

                            labels_detached = labels.cpu().detach().numpy()
                            pred_detached = pred.cpu().detach().numpy()
                            #print(f"Predicted: {pred.squeeze()}\nActual: {labels.squeeze()}")
                            trn_AUCs.append(roc_auc_score(labels_detached, pred_detached)) 
                            trn_precs.append(precision_score(labels_detached, pred_detached))
                            trn_recalls.append(recall_score(labels_detached, pred_detached))
                            trn_f1s.append(f1_score(labels_detached, pred_detached))
                            trn_losses.append(train_loss.item())
                            
                            trn_accs.append(accuracy_score(labels_detached, pred_detached))

                            avg_trn_acc = np.mean(trn_accs)
                            avg_trn_auc = np.mean(trn_AUCs)
                            avg_trn_prec = np.mean(trn_precs)
                            avg_trn_recall = np.mean(trn_recalls)
                            avg_trn_f1 = np.mean(trn_f1s)
                            avg_trn_loss = np.mean(trn_losses)


                        avg_val_loss, avg_val_auc, avg_val_acc, avg_val_precs, avg_val_recall, avg_val_f1 =val_test(val_loader)
                        
                        print(f"[Epoch: {epoch+1}] Average val AUC: {avg_val_auc}, ave acc: {avg_val_acc}")
                        if avg_val_auc > best_rnn:
                            # best RNN is based on the val AUC metric
                            best_rnn = avg_val_auc
                            best_trn_metrics = avg_trn_loss, avg_trn_auc, avg_trn_acc,  avg_trn_prec, avg_trn_recall, avg_trn_f1
                            best_val_metrics = avg_val_loss, avg_val_auc, avg_val_acc, avg_val_precs, avg_val_recall, avg_val_f1
                            
                            #best_rnn_combo = (nepochs, lr, layer1, layer2, layer3)
                            best_epochs = nepochs
                            best_lr = lr
                            best_layer1 = layer1
                            best_layer2 = layer2
                            best_layer3 = layer3
                            
model = RNNModel(hidden_dim = best_layer1, fc1_size=best_layer2, fc2_size=best_layer3)
model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device) # with logits loss does sigmoid
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
print("\nRunning best parameters on the test set...")
for epoch in range(best_epochs):    
    avg_tst_loss, avg_tst_auc, avg_tst_acc, avg_tst_precs, avg_tst_recall, avg_tst_f1 =val_test(test_loader)
    best_tst_metrics = avg_tst_loss, avg_tst_auc, avg_tst_acc, avg_tst_precs, avg_tst_recall, avg_tst_f1

    metric_order = ['loss', 'AUC', 'accuracy', 'precision', 'recall', 'F1']
#print(f"\nBest test auc: {best_rnn}")
print(f"\nBest params: epochs={best_epochs}, lr={best_lr}, layer1={best_layer1}, layer2={best_layer2}, layer3={best_layer3}")

def print_metrics(metrics='Training', best_metrics=best_trn_metrics):
    print(f'{metrics} metrics:')
    for metric_name,metric in zip(metric_order, best_metrics):
        print(f'\t {metric_name}: {metric}')
        
print_metrics()
print_metrics(metrics='Validation', best_metrics=best_val_metrics)
print_metrics(metrics='Testing', best_metrics=best_trn_metrics)







