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
# test_data  = './data/mini/test/'

device_type = 'cpu'
if torch.cuda.is_available():
    device_type = 'cuda'

# data processing
train_dataset, val_dataset = get_dataset(train_data, 10)

train_dl = DataLoader(train_dataset, batch_size = 1, shuffle=True)
val_dl   = DataLoader(val_dataset, batch_size = 5)
print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

# Debugging show few images
albedo, normal, mask, sh, face = next(iter(train_dl))
# save_image(albedo)
# save_image(normal)
# save_image(face)
# save_image(mask)
# masked_image = applyMask(face, mask)
# save_image(masked_image)

shading = getShadingFromNormalAndSH(normal, sh)
save_image(shading, mask=mask, path='./results/shading_from_normal/shading_ours.png')

recon   = shading * albedo
save_image(recon, mask=mask, path='./results/shading_from_normal/recon_ours.png')
save_image(face, mask=mask, path = './results/shading_from_normal/recon_groundtruth.png')

recon = applyMask(recon, mask)
face  = applyMask(face, mask)
mseLoss = nn.L1Loss()
print('L1Loss Ours: ', mseLoss(face, recon).item())

sfsnet_shading_net = sfsNetShading()
sh = sh.view(sh.shape[0], sh.shape[2])
sfs_shading = sfsnet_shading_net(normal, sh)
save_image(sfs_shading, mask=mask, path='./results/shading_from_normal/shading_sfsnet.png')
recon   = sfs_shading * albedo
save_image(recon, mask=mask, path='./results/shading_from_normal/recon_sfsnet.png')

recon = applyMask(recon, mask)
face  = applyMask(face, mask)
mseLoss = nn.L1Loss()
print('L1Loss SFSNet: ', mseLoss(face, recon).item())
