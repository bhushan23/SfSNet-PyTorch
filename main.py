import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

from data_loading import *
from utils import *
from shading import *
from train import *
from models import *

# initialization
train_data = '/home/bhushan/work/thesis/sfsnet/data/'

device_type = 'cpu'
use_cuda    = False
if torch.cuda.is_available():
    device_type = 'cuda'
    use_cuda    = True

# data processing
train_dataset, val_dataset = get_dataset(train_data, 10)

train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dl   = DataLoader(val_dataset, batch_size=1)
print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

# Debugging and check working
# validate_shading_method(train_dl)

# Initialize models

conv_model            = baseFeaturesExtractions()
normal_residual_model = NormalResidualBlock()
albedo_residual_model = AlbedoResidualBlock()
light_estimator_model = LightEstimator()
normal_gen_model      = NormalGenerationNet()
albedo_gen_model      = AlbedoGenerationNet()
shading_model         = sfsNetShading()
image_recon_model     = ReconstructImage()

if use_cuda:
    conv_model            = conv_model.cuda()
    normal_residual_model = normal_residual_model.cuda()
    albedo_residual_model = albedo_residual_model.cuda()
    light_estimator_model = light_estimator_model.cuda()
    normal_gen_model      = normal_gen_model.cuda()
    albedo_gen_model      = albedo_gen_model.cuda()
    shading_model         = shading_model.cuda()
    image_recon_model     = image_recon_model.cuda()

train(conv_model, normal_residual_model, albedo_residual_model, \
          light_estimator_model, normal_gen_model, albedo_gen_model, \
          shading_model, image_recon_model, train_dl, val_dl, \
          num_epochs = 1, log_path = './metadata/', use_cuda=use_cuda)
