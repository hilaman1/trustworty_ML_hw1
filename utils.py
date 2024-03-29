import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    total_accuracy = 0.0
    for batch, batch_labels in data_loader:
        # Transfer the data to device
        batch, batch_labels = batch.to(device), batch_labels.to(device)
        # forward pass
        batch_predictions = model(batch)
        # calculate accuracy
        predicted_labels = batch_predictions.argmax(dim=1, keepdim=True).squeeze()
        accuracy = torch.sum(predicted_labels == batch_labels).item() / batch.size(0)
        total_accuracy += accuracy
    return total_accuracy/len(data_loader)


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    x_adv_all = []
    y_all = []
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        if targeted:
            # Generate random target labels
            t = torch.remainder(y + torch.randint_like(y, 0, high=n_classes), n_classes)
            x_adv = attack.execute(x, t, targeted)
            y_all.append(t)
        else:
            # runs untargeted attacks
            x_adv = attack.execute(x, y, targeted)
            y_all.append(y)
        x_adv_all.append(x_adv)
    x_adv = torch.cat(x_adv_all)
    y = torch.cat(y_all)
    return x_adv, y

def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    num_queries_all = []
    x_adv_all = []
    y_all = []
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        if targeted:
            # Generate random target labels
            t = torch.remainder(y + torch.randint_like(y, 0, high=n_classes), n_classes)
            x_adv, num_queries = attack.execute(x, t, targeted)
            x_adv_all.append(x_adv)
            num_queries_all.append(num_queries)
            y_all.append(t)
        else:
            # runs untargeted attacks
            x_adv, num_queries = attack.execute(x, y, targeted)
            x_adv_all.append(x_adv)
            num_queries_all.append(num_queries)
            y_all.append(y)
    x_adv = torch.cat(x_adv_all)
    y = torch.cat(y_all)
    queries = torch.cat(num_queries_all)
    return x_adv, y, queries

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    total_accuracy = 0.0
    batch_num = 0
    for i in range(0, len(x_adv), batch_size):
        # Get batch of samples
        x_batch = x_adv[i:i + batch_size].to(device)
        y_batch = y[i:i + batch_size].to(device)
        # forward pass
        batch_predictions = model(x_batch)
        # calculate accuracy
        predicted_labels = batch_predictions.argmax(dim=1, keepdim=True).squeeze()
        if targeted:
            # Targeted attack
            accuracy = (predicted_labels == y_batch).float().mean().item()
        else:
            # Untargeted attack
            accuracy = (predicted_labels != y_batch).float().mean().item()
        total_accuracy += accuracy
        batch_num += 1
    return total_accuracy / batch_num


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass # FILL ME

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass # FILL ME

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass # FILL ME
