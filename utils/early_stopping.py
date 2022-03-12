#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_made = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1)
            self.checkpoint_made =True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
            self.checkpoint_made =False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1)
            self.counter = 0
            self.checkpoint_made =True
        
    def save_checkpoint(self, val_loss, model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def print_checkpoint_metric(self, val_loss, model, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_auc, val_batch_auc, test_batch_auc, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_prec, val_batch_prec, test_batch_prec, train_batch_recall, val_batch_recall, test_batch_recall, train_batch_f1, val_batch_f1, test_batch_f1, checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1):
        
        checkpoint_train_loss = np.mean(train_batch_losses)
        checkpoint_val_loss = np.mean(val_batch_loss)
        checkpoint_test_loss = np.mean(test_batch_loss)
        checkpoint_train_auc = np.mean(train_batch_auc)
        checkpoint_val_auc = np.mean(val_batch_auc)
        checkpoint_test_auc = np.mean(test_batch_auc)
        checkpoint_train_acc = np.mean(train_batch_acc)
        checkpoint_val_acc = np.mean(val_batch_acc)
        checkpoint_test_acc = np.mean(test_batch_acc)
        checkpoint_train_prec = np.mean(train_batch_prec)
        checkpoint_val_prec = np.mean(val_batch_prec)
        checkpoint_test_prec = np.mean(test_batch_prec)
        checkpoint_train_recall = np.mean(train_batch_recall)
        checkpoint_val_recall = np.mean(val_batch_recall)
        checkpoint_test_recall = np.mean(test_batch_recall)
        checkpoint_train_f1 = np.mean(train_batch_f1)
        checkpoint_val_f1 = np.mean(val_batch_f1)
        checkpoint_test_f1 = np.mean(test_batch_f1)
        return checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_auc, checkpoint_val_auc, checkpoint_test_auc, checkpoint_train_acc, checkpoint_val_acc, checkpoint_test_acc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1

