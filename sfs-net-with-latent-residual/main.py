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
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wt-decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    if ON_SERVER:
        train_csv = '/nfs/bigdisk/bsonawane/sfsnet_data/train.csv'
        test_csv  = '/nfs/bigdisk/bsonawane/sfsnet_data/test.csv'

        parser.add_argument('--train_data', type=str, default='/nfs/bigdisk/bsonawane/sfsnet_data/train/',
                        help='Training Dataset path')
        parser.add_argument('--test_data', type=str, default='/nfs/bigdisk/bsonawane/sfsnet_data/test/',
                        help='Testing Dataset path')
        parser.add_argument('--log_dir', type=str, default='./results/',
                        help='Log Path')
    else:  
        train_csv = '../data/train.csv'
        test_csv  = '../data/test.csv'
        parser.add_argument('--train_data', type=str, default='../data/train/',
                        help='Training Dataset path')
        parser.add_argument('--test_data', type=str, default='../data/test/',
                        help='Training Dataset path')
        parser.add_argument('--log_dir', type=str, default='./results/',
                        help='Log Path')

    parser.add_argument('--load_model', type=str, default=None,
                        help='load model from')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # initialization
    train_data = args.train_data
    test_data  = args.test_data
    batch_size = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    model_dir  = args.load_model

    # data processing
    train_dataset, val_dataset = get_dataset(train_data, read_from_csv=train_csv, validation_split=10)
    test_dataset, _ = get_dataset(test_data, read_from_csv=test_csv, validation_split=0)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

    # Init WandB for logging
    wandb.init(project='SfSNet-Residual-From-Latent-Light')
    wandb.log({'lr':lr, 'weight decay': wt_decay})

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
    neural_light_model    = NeuralLatentLightEstimator()
    shading_correctness_model = ShadingCorrectNess()
    
    sfs_net_pipeline      = SfsNetPipeline(conv_model, normal_residual_model, albedo_residual_model, \
                                            light_estimator_model, normal_gen_model, albedo_gen_model, \
                                            shading_model, neural_light_model, shading_correctness_model, image_recon_model)
    if use_cuda:
        sfs_net_pipeline = sfs_net_pipeline.cuda()

    if model_dir is not None:
        sfs_net_pipeline.load_state_dict(torch.load(model_dir + 'sfs_net_model.pkl'))

    wandb.watch(sfs_net_pipeline)
    train(sfs_net_pipeline, train_dl, val_dl, test_dl, \
            num_epochs=epochs, log_path=log_dir, use_cuda=use_cuda, wandb=wandb, \
            lr=lr, wt_decay=wt_decay)

if __name__ == '__main__':
    main()
