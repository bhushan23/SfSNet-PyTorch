import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loading import *
from utils import *
from shading import *

# initialization
train_data = '/home/bhushan/work/thesis/sfsnet/data/'
# test_data  = './data/mini/test/'

device_type = 'cpu'
if torch.cuda.is_available():
    device_type = 'cuda'

# data processing
train_dataset, val_dataset = get_dataset(train_data, 10)

train_dl = DataLoader(train_dataset, batch_size = 2, shuffle=True)
val_dl   = DataLoader(val_dataset, batch_size = 5)
print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

# Debugging show few images
albedo, normal, mask, sh, face = next(iter(train_dl))
print(albedo.shape)
# save_image(albedo)
# save_image(normal)
# save_image(face)
# save_image(mask)
# masked_image = applyMask(face, mask)
# save_image(masked_image)

shading = getShadingFromNormalAndSH(normal, sh)
save_image(shading, mask=mask)
print(shading.shape)

