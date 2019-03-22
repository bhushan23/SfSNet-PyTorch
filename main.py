import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse

import wandb
from data_loading import *
from utils import *
from shading import *
from train import *
from models import *

def main():
    ON_SERVER = False

    parser = argparse.ArgumentParser(description='SfSNet - Residual')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wt-decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    if ON_SERVER:
        parser.add_argument('--train_data', type=str, default='/home/bhushan/work/thesis/sfsnet/data/',
                        help='Training Dataset path')
        parser.add_argument('--log_dir', type=str, default='/home/bhushan/work/thesis/sfsnet/results/',
                        help='Log Path')
    else:  
        parser.add_argument('--train_data', type=str, default='./data/',
                        help='Training Dataset path')
        parser.add_argument('--log_dir', type=str, default='./results/',
                        help='Log Path')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # initialization
    train_data = args.train_data
    batch_size = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt-decay
    log_dir    = args.log_dir

    wandb.log({'lr':lr, 'weight decay': wt_decay})

    # data processing
    train_dataset, val_dataset = get_dataset(train_data, 10)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

    # Init WandB for logging
    wandb.init(project='SfSNet-Base')

    # Debugging and check working
    # validate_shading_method(train_dl, wandb)

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
            num_epochs=10, log_path=log_dir, use_cuda=use_cuda, wandb=wandb, \
            lr=lr, wt_decay=wt_decay)

if __name__ == '__main__':
    main()