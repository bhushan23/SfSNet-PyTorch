import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

from data_loading import *
from utils import *
from shading import *

# initialization
train_data = '/home/bhushan/work/thesis/sfsnet/data/'

device_type = 'cpu'
if torch.cuda.is_available():
    device_type = 'cuda'
    use_cuda    = True

# data processing
train_dataset, val_dataset = get_dataset(train_data, 10)

train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dl   = DataLoader(val_dataset, batch_size=5)
print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

# Debugging and check working
validate_shading_method(train_dl)

