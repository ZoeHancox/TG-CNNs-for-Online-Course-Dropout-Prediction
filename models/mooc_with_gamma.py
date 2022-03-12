import numpy as np
import scipy as sp
import pandas as pd
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torchvision
import torchvision.models as models

from torch.utils.data import TensorDataset, DataLoader, random_split


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import random
import sys
import time
from datetime import timedelta
start_time = time.monotonic()


actions_to_use = "100" 
total_targetIDs = 97
debug = False


sample_size = 7040
filename = str(sys.argv[0])[:-3] # filename minus .py

gamma = 1
in_chans = 1 #input_channels

random.seed(time.time())


#---------------------
checkpoint_path = filename+'_checkpoint.pt'
metric_values = "metric_vals"+filename+".npy"
loss_fig = "loss_plot_"+filename+".png"
# ---------------------
from utils/early_stopping import EarlyStopping 

checkpoint_train_loss = 0.0
checkpoint_val_loss = 0.0
checkpoint_test_loss = 0.0
checkpoint_train_auc = 0.0
checkpoint_val_auc = 0.0
checkpoint_test_auc = 0.0
checkpoint_train_acc = 0.0
checkpoint_val_acc = 0.0
checkpoint_test_acc = 0.0
checkpoint_train_prec = 0.0
checkpoint_val_prec = 0.0
checkpoint_test_prec = 0.0
checkpoint_train_recall = 0.0
checkpoint_val_recall = 0.0
checkpoint_test_recall = 0.0
checkpoint_train_f1 = 0.0
checkpoint_val_f1 = 0.0
checkpoint_test_f1 = 0.0


###################Random Hyperparameter Selection ####################
epoch_hyperparams = [25, 50, 75, 100]
nepochs = random.choice(epoch_hyperparams)

lr_hyperparams = [0.01, 0.05, 0.001, 0.005, 0.0001]
lr = random.choice(lr_hyperparams)

filter_hyperparams = [32, 64, 128]
out_chans = random.choice(filter_hyperparams)

filter_size_hyperparams = [4, 16, 32, 64]
num_time_steps = random.choice(filter_size_hyperparams)

lstm_hyperparams = [16, 32, 64, 128, 256]
lstm_h = random.choice(lstm_hyperparams)

weight_decay_hyperparams = [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5]
weight_decay_rand = random.choice(weight_decay_hyperparams)

fcl_size = [128, 256, 512, 1028, 2056] # linear layer output size
linear_size = random.choice(fcl_size)

drop_vals = [0.2, 0.3, 0.4, 0.5]
drop_val = random.choice(drop_vals)



batch_sizes = [32, 64, 128]
val_batch_size = 128
test_batch_size = 128
train_batch_size = random.choice(batch_sizes)


 #################### #################### #################### ########

print("="*40)
print("HYPERPARAMETERS\n\t\t---")
print("Filename:", filename)
print(f"Epochs: {nepochs}")
print(f"Learning rate: {lr}")
print(f"Filters: {out_chans}")
print(f"Filter size: {num_time_steps}")
print(f"Hidden LSTM neurons: {lstm_h}")
print(f"Weight decay (L2 regularisation): {weight_decay_rand}")
print(f"Fully connected layer output size: {linear_size}")
print(f"Dropout value: {drop_val}")
print(f"Batch sizes = trn:{train_batch_size}, val:{val_batch_size}, tst:{test_batch_size}")
print("="*40)

"""## Loading the dataframe"""

if actions_to_use == "all":
    targetid_time_df = pd.read_pickle("data/processed_data/targetid_and_scaled_time_all.pkl")
elif actions_to_use == "100":
    targetid_time_df = pd.read_pickle("data/processed_data/targetid_and_scaled_time_last100.pkl")

# to check the number of actions taken
column = targetid_time_df['TARGETID']

num_of_actions = []
for row in column:
    num_of_actions.append(len(row))

max_action = max(num_of_actions)
print("Most actions taken by one user:", max_action)
targetid_time_df.head()

"""## Making the sparse matrices from the dataframe"""

users = len(targetid_time_df.index) # number of rows in df
print("Number of users:", users)

input_matrices = []

values_lists = []
indices_lists = []


for user in range(sample_size):
    
    # target and time list relevant to each user, one by one
    current_target_list = targetid_time_df.iloc[user]["TARGETID"]
    current_time_list = targetid_time_df.iloc[user]["SCALED_TIMESTAMP"]

    unique_tID = len(set(current_target_list)) # num of unique target IDs
    num_time_jumps = len(current_time_list)-1 # num of time jumps (minus 1 to ignore 0)

    # finding the time difference between each target action
    time_jump_list = []

    for first, second in zip(current_time_list, current_time_list[1:]):
        diff = second-first
        time_jump_list.append(diff)
    
    i_list = []
    v_list = []
    for node_start, node_end, time_jump_num, time_jump in zip(current_target_list, current_target_list[1:], 
                                                              range(num_time_jumps), time_jump_list):       
        
        # max_action - num_time_jumps difference moves the time_jumps to the end so padding is at front
        i = [(max_action-num_time_jumps)+time_jump_num, node_start, node_end] #indices for coordinates
        i_list.append(i)
        v = int(time_jump) #values
        v_list.append(v)
        
    values_lists.append(v_list)
    indices_lists.append(i_list)
    
    s1 = torch.sparse_coo_tensor(torch.tensor(i_list).t(), 
                                v_list, (max_action, 97, 97),
                               dtype=torch.int64)
    
    input_matrices.append(s1)
    

"""## For the output / labels and batch sizes and spliting"""

random.seed(1)
labels_file = 'data/processed_data/labels.npy'

loaded_y = np.load(labels_file) # gives as numpy array
loaded_y = loaded_y[:sample_size]

# needs converting to a list when loaded
y = loaded_y.tolist()

train_split = round(0.8*sample_size)
val_split = sample_size-train_split


data = []
for i in range(sample_size):
    data.append([input_matrices[i], y[i]])

train_dataset, val_dataset = random_split(data, [train_split, val_split])

test_split = int(len(val_dataset)/2)
val_dataset, test_dataset = random_split(val_dataset, [test_split, test_split])

train_loader = DataLoader(dataset = train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(dataset = val_dataset, batch_size=val_batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=test_batch_size, shuffle=True)




import torch, torch.nn as nn

"""### Sparse to dense function"""

def sparse_to_dense(matrix):
    """
    Transforms a list of sparse coo matrices into a list of dense arrays
    Converts to a tensor for input into the model
    """
    dense = [] # list of dense tensors
    for sparse in matrix:
        s = sparse.to_dense() # reverts sparse back to dense tensor
        dense.append(s.numpy())
    dense = torch.Tensor(dense)
    return dense

"""### Validation function"""

def validation(loader=val_loader):
    val_losses, val_accs, val_AUCs, val_precs, val_recalls, val_f1s = ([] for i in range(6))


    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
      
            inputs = sparse_to_dense(inputs)

            val_inputs = inputs.to(device).unsqueeze(1)

            labels = labels.to(device)
            val_labels = labels.unsqueeze(1).float()

            val_out = model(val_inputs.float())
            val_loss = criterion(val_out, val_labels)

            val_losses.append(val_loss.item()) # list all val_losses over the batch
            

            sig = nn.Sigmoid()
            pred = sig(val_out).round()
            pred = torch.transpose(pred, 0, 1).squeeze()

            val_accuracy = (pred == labels).double().mean().item()
            val_accs.append(val_accuracy) # list all val_accs over the batch
            
 
            val_auc = roc_auc_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
            val_AUCs.append(val_auc)

            val_prec = precision_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy(), zero_division=True)
            val_precs.append(val_prec)

            val_recall = recall_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
            val_recalls.append(val_recall)

            val_f1 = f1_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
            val_f1s.append(val_f1)

    return val_losses, val_accs, val_AUCs, val_precs, val_recalls, val_f1s

"""### Model network"""

# input size = max_actions x 97 x 97
conv_D_out = max_action - (num_time_steps - 1) # so long as stride=1 and padding =0
# in_feats = conv3d_out * out_chans

class Dense3DConv(nn.Module):
    def __init__(self):
        super(Dense3DConv, self).__init__()
        self.conv3d = nn.Conv3d(in_chans, out_chans, kernel_size=(num_time_steps,97,97), stride=1, bias=False) 
        self.batchnorm = nn.BatchNorm3d(out_chans)
        self.lstm = nn.LSTM(input_size = out_chans, hidden_size= lstm_h, batch_first = True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(conv_D_out * lstm_h, linear_size) #  conv_out * hidden_size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(linear_size, 1) # input chans, number of labels
        
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=drop_val)
    

    def forward(self, x):
        x = x.to(device)
        scaled_matrix = torch.where(x==0, x, torch.exp(-gamma*x)) # condition for only non-zero elements
        out = self.conv3d(scaled_matrix)#.squeeze(0) # squeeze removes all one dimensional parts from the tensor
        
        out = self.batchnorm(out)
        
        out = torch.squeeze(out, 4)
        out = torch.squeeze(out, 3)
        
        out = self.relu(out)
        
        out, cell_state = self.lstm(torch.transpose(out, 1, 2))
        
        out = self.flat(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = Dense3DConv().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=weight_decay_rand)
criterion = nn.BCEWithLogitsLoss().to(device)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=True)
early_stopping = EarlyStopping(patience=50, path= checkpoint_path, verbose=True)

model = model.float()

#Set seeds
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)


"""### Training"""

print("\n-Model layers-\n",model)
###for graphs
avg_train_batch_losses, avg_val_batch_losses, avg_test_batch_losses = [], [], []
avg_val_batch_auc, avg_train_batch_auc, avg_test_batch_auc = [], [], []
avg_val_batch_acc, avg_train_batch_acc, avg_test_batch_acc = [], [], []
avg_val_batch_prec, avg_train_batch_prec, avg_test_batch_prec = [], [], []
avg_val_batch_recall, avg_train_batch_recall, avg_test_batch_recall = [], [], []
avg_val_batch_f1, avg_train_batch_f1, avg_test_batch_f1 = [], [], []
###

print('Starting training...')
for epoch in range(nepochs):
    train_batch_losses = []
    train_batch_acc = []
    train_batch_auc =[]
    train_batch_prec = []
    train_batch_recall = []
    train_batch_f1 = []
    for i, batch in enumerate(train_loader):
        matrix, labels = batch
        dense = sparse_to_dense(matrix)

        inputs = dense.to(device).unsqueeze(1) # adds another dimension to the front for channel size
        labels = labels.to(device) # this is for accuracy calc
        targets = labels.unsqueeze(1).float() # to get the same dims as input

        model.train()

        outputs = model(inputs.float())
        train_loss = criterion(outputs, targets) # loss function

        train_loss.backward()
        optimizer.step() # update the weights
        optimizer.zero_grad() # zero parameter gradients to prevent carrying forward values from previous iter
       
        
        sig = nn.Sigmoid()
        pred = sig(outputs).round()
        pred = torch.transpose(pred, 0, 1).squeeze()

        train_batch_losses.append(train_loss.item())
        train_accuracy = (pred == labels).double().mean().item()
        train_batch_acc.append(train_accuracy)
        
        AUC = roc_auc_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        train_batch_auc.append(AUC)

        trn_precision = precision_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy(),  zero_division=True)
        train_batch_prec.append(trn_precision)

        trn_recall = recall_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        train_batch_recall.append(trn_recall)

        trn_f1 = f1_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        train_batch_f1.append(trn_f1)

        if debug:
            print(f"epoch [{epoch+1}/{nepochs}], step [{i+1}/{len(train_loader)}], training loss {train_loss.item():.5f}")
            print("TRAINING SCORES\nAUC Score:", AUC)
            print("Precision Score:", trn_precision)
            print("Recall Score:", trn_recall)
            print("F1 Score:", trn_f1)
  
            print('---')


    val_batch_loss, val_batch_acc, val_batch_auc, val_batch_prec, val_batch_recall, val_batch_f1 = validation(val_loader)
    test_batch_loss, test_batch_acc, test_batch_auc, test_batch_prec, test_batch_recall, test_batch_f1 = validation(test_loader)
  
    def metric_save(trn_epoch_metric=train_batch_losses, trn_all_epoch_avgs=avg_train_batch_losses, 
                    val_epoch_metric=val_batch_loss, val_all_epoch_avgs=avg_val_batch_losses,#):
                    test_epoch_metric=test_batch_loss, test_all_epoch_avgs=avg_test_batch_losses):
        trn_all_epoch_avgs.append(np.mean(trn_epoch_metric))
        val_all_epoch_avgs.append(np.mean(val_epoch_metric))  
        test_all_epoch_avgs.append(np.mean(test_epoch_metric))


    metric_save() #losses
    metric_save(train_batch_auc, avg_train_batch_auc, val_batch_auc, avg_val_batch_auc, test_batch_auc, avg_test_batch_auc) # auc
    metric_save(train_batch_acc, avg_train_batch_acc, val_batch_acc, avg_val_batch_acc, test_batch_acc, avg_test_batch_acc) # accuracy
    metric_save(train_batch_prec, avg_train_batch_prec, val_batch_prec, avg_val_batch_prec, test_batch_prec, avg_test_batch_prec) # prec
    metric_save(train_batch_recall, avg_train_batch_recall, val_batch_recall, avg_val_batch_recall, test_batch_recall, avg_test_batch_recall) # recall
    metric_save(train_batch_f1, avg_train_batch_f1, val_batch_f1, avg_val_batch_f1, test_batch_f1, avg_test_batch_f1) # f1



    scheduler.step(np.mean(val_batch_loss)) # to alter the learning rate

    print(f'[Epoch {epoch+1}] \n\t\tTrain\t\t Validation\t\t Test\nLoss: \t\t{np.mean(train_batch_losses):.4f} \t\t {np.mean(val_batch_loss):.4f} \t\t {np.mean(test_batch_loss):.4f}')
    print(f"Accuracy:\t{np.mean(train_batch_acc): .3%} \t {np.mean(val_batch_acc): .3%} \t\t {np.mean(test_batch_acc): .3%}")
    print(f"AUC: \t\t{np.mean(train_batch_auc):.4f} \t\t {np.mean(val_batch_auc):.4f} \t\t {np.mean(test_batch_auc):.4f}")

    print(f"Precision: \t{np.mean(train_batch_prec):.4f} \t\t {np.mean(val_batch_prec):.4f} \t\t {np.mean(test_batch_prec):.4f}")
    print(f"Recall: \t{np.mean(train_batch_recall):.4f} \t\t {np.mean(val_batch_recall):.4f} \t\t {np.mean(val_batch_recall):.4f}")
    print(f"F1: \t\t{np.mean(train_batch_f1):.4f} \t\t {np.mean(val_batch_f1):.4f} \t\t {np.mean(test_batch_f1):.4f}")
    
        
    early_stopping(np.mean(val_batch_loss), model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1)
    
    if early_stopping.checkpoint_made:
        checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1 = early_stopping.print_checkpoint_metric(np.mean(val_batch_loss), model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1)

    
    
    print("="*30)

    
    if early_stopping.early_stop:
        print("Early stopping")
        break


    
print('Training complete!')

print("***"*10)
print("HYPERPARAMETERS\n\t\t---")
print(filename)
print(f"Epochs: {nepochs}")
print(f"Learning rate: {lr}")
print(f"Filters: {out_chans}")
print(f"Filter size: {num_time_steps}")
print(f"Hidden LSTM neurons: {lstm_h}")
print(f"Weight decay (L2 regularisation): {weight_decay_rand}")
print(f"Fully connected layer output size: {linear_size}")
print(f"Dropout value: {drop_val}")
print(f"Batch sizes = trn:{train_batch_size}, val:{val_batch_size}, tst:{test_batch_size}")

print("=="*6, 'METRIC_RANGES', "=="*6)

print(f"\nLOSS\nTrain range: \t\t{min(avg_train_batch_losses):.4f}-{max(avg_train_batch_losses):.4f}")
print(f"Validation range: \t{min(avg_val_batch_losses):.4f}-{max(avg_val_batch_losses):.4f}")
print(f"Test range: \t\t{min(avg_test_batch_losses):.4f}-{max(avg_test_batch_losses):.4f}")

print(f"\nAUC\nTrain range: \t\t{min(avg_train_batch_auc):.4f}-{max(avg_train_batch_auc):.4f}")
print(f"Validation range: \t{min(avg_val_batch_auc):.4f}-{max(avg_val_batch_auc):.4f}")
print(f"Test range: \t\t{min(avg_test_batch_auc):.4f}-{max(avg_test_batch_auc):.4f}")

print(f"\nACCURACY\nTrain range: \t\t{min(avg_train_batch_acc):.2%}-{max(avg_train_batch_acc):.2%}")
print(f"Validation range: \t{min(avg_val_batch_acc):.2%}-{max(avg_val_batch_acc):.2%}")
print(f"Test range: \t\t{min(avg_test_batch_acc):.2%}-{max(avg_test_batch_acc):.2%}")

print(f"\nPRECISION\nTrain range: \t\t{min(avg_train_batch_prec):.4f}-{max(avg_train_batch_prec):.4f}")
print(f"Validation range: \t{min(avg_val_batch_prec):.4f}-{max(avg_val_batch_prec):.4f}")
print(f"Test range: \t\t{min(avg_test_batch_prec):.4f}-{max(avg_test_batch_prec):.4f}")

print(f"\nRECALL\nTrain range: \t\t{min(avg_train_batch_recall):.4f}-{max(avg_train_batch_recall):.4f}")
print(f"Validation range: \t{min(avg_val_batch_recall):.4f}-{max(avg_val_batch_recall):.4f}")
print(f"Test range: \t\t{min(avg_test_batch_recall):.4f}-{max(avg_test_batch_recall):.4f}")

print(f"\nF1\nTrain range: \t\t{min(avg_train_batch_f1):.4f}-{max(avg_train_batch_f1):.4f}")
print(f"Validation range: \t{min(avg_val_batch_f1):.4f}-{max(avg_val_batch_f1):.4f}")
print(f"Test range: \t\t{min(avg_test_batch_f1):.4f}-{max(avg_test_batch_f1):.4f}\n")


end_time = time.monotonic()
time_taken = timedelta(seconds=end_time - start_time)
print(time_taken)


print("Copy and paste below into excel:")
print(f"{filename}\t{nepochs}\t{epoch+1}\t{str(time_taken)[0:4].replace(':','h')}m\t{lr}\t{out_chans}",
      f"\t{num_time_steps}\t{lstm_h}\t{weight_decay_rand}\t{linear_size}\t{drop_val}\t{train_batch_size}\t{val_batch_size}\t{test_batch_size}",

      f"\t{min(avg_train_batch_losses)}\t{max(avg_train_batch_losses)}\t{min(avg_val_batch_losses)}\t{max(avg_val_batch_losses)}\t{min(avg_test_batch_losses)}\t{max(avg_test_batch_losses)}",
      f"\t{checkpoint_train_loss}\t{checkpoint_val_loss}\t{checkpoint_test_loss}",

      f"\t{min(avg_train_batch_auc)}\t{max(avg_train_batch_auc)}\t{min(avg_val_batch_auc)}\t{max(avg_val_batch_auc)}\t{min(avg_test_batch_auc)}\t{max(avg_test_batch_auc)}",
      f"\t{checkpoint_train_auc}\t{checkpoint_val_auc}\t{checkpoint_test_auc}",

      f"\t{min(avg_train_batch_acc)}\t{max(avg_train_batch_acc)}\t{min(avg_val_batch_acc)}\t{max(avg_val_batch_acc)}\t{min(avg_test_batch_acc)}\t{max(avg_test_batch_acc)}",
      f"\t{checkpoint_train_acc}\t{checkpoint_val_acc}\t{checkpoint_test_acc}",

      f"\t{min(avg_train_batch_prec)}\t{max(avg_train_batch_prec)}\t{min(avg_val_batch_prec)}\t{max(avg_val_batch_prec)}\t{min(avg_test_batch_prec)}\t{max(avg_test_batch_prec)}",
      f"\t{checkpoint_train_prec}\t{checkpoint_val_prec}\t{checkpoint_test_prec}",

      f"\t{min(avg_train_batch_recall)}\t{max(avg_train_batch_recall)}\t{min(avg_val_batch_recall)}\t{max(avg_val_batch_recall)}\t{min(avg_test_batch_recall)}\t{max(avg_test_batch_recall)}",
      f"\t{checkpoint_train_recall}\t{checkpoint_val_recall}\t{checkpoint_test_recall}",

      f"\t{min(avg_train_batch_f1)}\t{max(avg_train_batch_f1)}\t{min(avg_val_batch_f1)}\t{max(avg_val_batch_f1)}\t{min(avg_test_batch_f1)}\t{max(avg_test_batch_f1)}",
      f"\t{checkpoint_train_f1}\t{checkpoint_val_f1}\t{checkpoint_test_f1}",

      f"\t{gamma}")





#model.load_state_dict(torch.load(checkpoint_path))

with open(metric_values, 'wb') as f:
    np.save(f, np.array(avg_train_batch_losses))
    np.save(f, np.array(avg_val_batch_losses))
    np.save(f, np.array(avg_test_batch_losses))

    np.save(f, np.array(avg_val_batch_acc))
    np.save(f, np.array(avg_train_batch_acc))
    np.save(f, np.array(avg_test_batch_acc))

    np.save(f, np.array(avg_train_batch_auc))
    np.save(f, np.array(avg_val_batch_auc))
    np.save(f, np.array(avg_test_batch_auc))

    np.save(f, np.array(avg_train_batch_prec))
    np.save(f, np.array(avg_val_batch_prec))
    np.save(f, np.array(avg_test_batch_prec))

    np.save(f, np.array(avg_train_batch_recall))
    np.save(f, np.array(avg_val_batch_recall))
    np.save(f, np.array(avg_test_batch_recall))

    np.save(f, np.array(avg_train_batch_f1))
    np.save(f, np.array(avg_val_batch_f1))
    np.save(f, np.array(avg_test_batch_f1))

    