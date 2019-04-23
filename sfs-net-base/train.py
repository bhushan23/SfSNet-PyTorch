import torch
import torch.nn as nn
import numpy as np
import os

from models import *
from utils import *
from data_loading import *

## TODOS:
## 1. Dump SH in file
## 
## 
## Notes:
## 1. SH is not normalized
## 2. Face is normalized and denormalized - shall we not normalize in the first place?


# Enable WANDB Logging
WANDB_ENABLE = True

def predict_celeba(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'CelebA_Val', dump_all_images = False):
 
    # debugging flag to dump image
    fix_bix_dump = 0
    recon_loss  = nn.L1Loss() 

    if use_cuda:
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    rloss = 0 # Reconstruction loss

    for bix, data in enumerate(dl):
        face = data
        if use_cuda:
            face   = face.cuda()
        
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)
        
        if bix == fix_bix_dump or dump_all_images:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(bix)
            # log images
            wandb_log_images(wandb, predicted_normal, None, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, None, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, predicted_shading, None, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, predicted_face, None, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, None, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth', path=file_name + '_gt_face.png')

            # TODO:
            # Dump SH as CSV or TXT file
        
        # Loss computation
        # Reconstruction loss
        total_loss  = recon_loss(predicted_face, face)

        # Logging for display and debugging purposes
        tloss += total_loss.item()
    
    len_dl = len(dl)
    wandb.log({suffix+' Total loss': tloss/len_dl}, step=train_epoch_num)
            

    # return average loss over dataset
    return tloss / len_dl

def predict_sfsnet(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image
    fix_bix_dump = 0

    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.L1Loss() 

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    nloss = 0 # Normal loss
    aloss = 0 # Albedo loss
    shloss = 0 # SH loss
    rloss = 0 # Reconstruction loss
    for bix, data in enumerate(dl):
        albedo, normal, mask, sh, face = data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            mask   = mask.cuda()
            sh     = sh.cuda()
            face   = face.cuda()
        
        # Apply Mask on input image
        # face = applyMask(face, mask)
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)

        if bix == fix_bix_dump:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
            # log images
            save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)

            wandb_log_images(wandb, save_p_normal, mask, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, predicted_shading, mask, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, predicted_face, mask, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, mask, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, suffix+' Ground Truth Normal', train_epoch_num, suffix+' Ground Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, suffix+' Ground Truth Albedo', train_epoch_num, suffix+' Ground Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real SH
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, predicted_albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Val Real SH Predicted Face', train_epoch_num, 'Val Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, syn_face, mask, 'Val Real SH GT Face', train_epoch_num, 'Val Real SH GT Face', path=file_name + '_syn_gt_face.png')

            # TODO:
            # Dump SH as CSV or TXT file
        
        # Loss computation
        # Normal loss
        current_normal_loss = normal_loss(predicted_normal, normal)
        # Albedo loss
        current_albedo_loss = albedo_loss(predicted_albedo, albedo)
        # SH loss
        current_sh_loss     = sh_loss(predicted_sh, sh)
        # Reconstruction loss
        current_recon_loss  = recon_loss(predicted_face, face)

        total_loss = lamda_recon * current_recon_loss + lamda_normal * current_normal_loss \
                        + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss

        # Logging for display and debugging purposes
        tloss += total_loss.item()
        nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()
    
    len_dl = len(dl)
    wandb.log({suffix+' Total loss': tloss/len_dl, 'Val Albedo loss': aloss/len_dl, 'Val Normal loss': nloss/len_dl, \
               'Val SH loss': shloss/len_dl, 'Val Recon loss': rloss/len_dl}, step=train_epoch_num)
            
    # return average loss over dataset
    return tloss / len_dl, nloss / len_dl, aloss / len_dl, shloss / len_dl, rloss / len_dl

def train(sfs_net_model, syn_data, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = None
    celeba_test_csv = None
    if celeba_data is not None:
        celeba_train_csv = celeba_data + '/train.csv'
        celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv, read_first=read_first, validation_split=2)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv, read_first=100, validation_split=0)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss(reduction='sum')
    recon_loss  = nn.L1Loss() 

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    syn_train_len    = len(syn_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        nloss = 0 # Normal loss
        aloss = 0 # Albedo loss
        shloss = 0 # SH loss
        rloss = 0 # Reconstruction loss

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask   = mask.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
           
            # Apply Mask on input image
            # face = applyMask(face, mask)
            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net_model(face)
            
            # Loss computation
            # Normal loss
            current_normal_loss = normal_loss(predicted_normal, normal)
            # Albedo loss
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            # SH loss
            current_sh_loss     = sh_loss(predicted_sh, sh)
            # Reconstruction loss
            # Edge case: Shading generation requires denormalized normal and sh
            # Hence, denormalizing face here
            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_normal * current_normal_loss \
                           + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss + lamda_recon * current_recon_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()

        print('Epoch: {} - Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(epoch, tloss, \
                                                                                                    nloss, aloss, shloss, rloss))
        log_prefix = 'Syn Data'
        if celeba_data is not None:
            log_prefix = 'Mix Data '

        if epoch % 1 == 0:
            print('Training set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(tloss / syn_train_len, \
                    nloss / syn_train_len, aloss / syn_train_len, shloss / syn_train_len, rloss / syn_train_len))
            # Log training info
            wandb.log({log_prefix + 'Train Total loss': tloss/syn_train_len, log_prefix + 'Train Albedo loss': aloss/syn_train_len, log_prefix + 'Train Normal loss': nloss/syn_train_len, \
                       log_prefix + 'Train SH loss': shloss/syn_train_len, log_prefix + 'Train Recon loss': rloss/syn_train_len})
            
            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' +  'train_' + str(epoch)
            save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            wandb_log_images(wandb, save_p_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', path=file_name + '_predicted_face.png')
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real_sh, predicted normal and albedo for debugging
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, predicted_albedo)
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Train Real SH Predicted Face', epoch, 'Train Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            wandb_log_images(wandb, syn_face, mask, 'Train Real SH GT Face', epoch, 'Train Real SH GT Face', path=file_name + '_syn_gt_face.png')

            v_total, v_normal, v_albedo, v_sh, v_recon = predict_sfsnet(sfs_net_model, syn_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                         out_folder=out_syn_images_dir+'/val/', wandb=wandb)
            wandb.log({log_prefix + 'Val Total loss': v_total, log_prefix + 'Val Albedo loss': v_albedo, log_prefix + 'Val Normal loss': v_normal, \
                        log_prefix + 'Val SH loss': v_sh, log_prefix + 'Val Recon loss': v_recon})
            

            print('Val set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(v_total,
                    v_normal, v_albedo, v_sh, v_recon))
            
            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
        if epoch % 5 == 0:
            t_total, t_normal, t_albedo, t_sh, t_recon = predict_sfsnet(sfs_net_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='Test')

            wandb.log({log_prefix+'Test Total loss': t_total, log_prefix+'Test Albedo loss': t_albedo, log_prefix+'Test Normal loss': t_normal, \
                       log_prefix+ 'Test SH loss': t_sh, log_prefix+'Test Recon loss': t_recon})

            print('Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}\n'.format(t_total,
                                                                                                    t_normal, t_albedo, t_sh, t_recon))

def train_syn_celeba_both(sfs_net_model, syn_data, celeba_data,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    celeba_train_csv = celeba_data + '/train.csv'
    celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(dir=syn_data+'train/', read_from_csv=syn_train_csv, validation_split=10)
    test_dataset, _ = get_sfsnet_dataset(dir=syn_data+'test/', read_from_csv=syn_test_csv, validation_split=0)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Load CelebA dataset
    train_dataset, val_dataset = get_celeba_dataset(read_from_csv=celeba_train_csv, validation_split=10)
    test_dataset, _ = get_celeba_dataset(read_from_csv=celeba_test_csv, validation_split=0)

    celeba_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    celeba_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    celeba_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))
    print('CelebA dataset: Train data: ', len(celeba_train_dl), ' Val data: ', len(celeba_val_dl), ' Test data: ', len(celeba_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir + 'syn/'
    out_celeba_images_dir = out_images_dir + 'celeba/'

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
    os.system('mkdir -p {}'.format(out_celeba_images_dir  + 'train/'))
    os.system('mkdir -p {}'.format(out_celeba_images_dir  + 'val/'))
    os.system('mkdir -p {}'.format(out_celeba_images_dir  + 'test/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.L1Loss() 
    c_recon_loss = nn.L1Loss() 

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()
        c_recon_loss = c_recon_loss.cuda()

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    syn_train_len    = len(syn_train_dl)
    celeba_train_len  = len(celeba_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        nloss = 0 # Normal loss
        aloss = 0 # Albedo loss
        shloss = 0 # SH loss
        rloss = 0 # Reconstruction loss
        celeba_tloss = 0 # Celeba Total loss

        # Initiate iterators
        syn_train_iter    = iter(syn_train_dl)
        celeba_train_iter = iter(celeba_train_dl)
        # Until we process both Synthetic and CelebA data
        while True:
            # Get and train on Synthetic dataset
            data = next(syn_train_iter, None)
            if data is not None:
                albedo, normal, mask, sh, face = data
                if use_cuda:
                    albedo = albedo.cuda()
                    normal = normal.cuda()
                    mask   = mask.cuda()
                    sh     = sh.cuda()
                    face   = face.cuda()
                
                # Apply Mask on input image
                face = applyMask(face, mask)

                predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net_model(face)

                # Loss computation
                # Normal loss
                current_normal_loss = normal_loss(predicted_normal, normal)
                # Albedo loss
                current_albedo_loss = albedo_loss(predicted_albedo, albedo)
                # SH loss
                current_sh_loss     = sh_loss(predicted_sh, sh)
                # Reconstruction loss
                # Edge case: Shading generation requires denormalized normal and sh
                # Hence, denormalizing face here
                current_recon_loss  = recon_loss(out_recon, denorm(face))

                total_loss = lamda_recon * current_recon_loss + lamda_normal * current_normal_loss \
                                + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()

                # Logging for display and debugging purposes
                tloss += total_loss.item()
                nloss += current_normal_loss.item()
                aloss += current_albedo_loss.item()
                shloss += current_sh_loss.item()
                rloss += current_recon_loss.item()

            # Get and train on CelebA data
            c_data = next(celeba_train_iter, None)
            if c_data is not None:
                # Get Mask as well if available
                c_mask = None
                if use_cuda:
                    c_data   = c_data.cuda()
                
                c_face = c_data
                # Apply Mask on input image
                # face = applyMask(face, mask)

                c_predicted_normal, c_predicted_albedo, c_predicted_sh, c_out_shading, c_out_recon = sfs_net_model(c_face)

                # Loss computation
                # Reconstruction loss
                # Edge case: Shading generation requires denormalized normal and sh
                # Hence, denormalizing face here
                crecon_loss = c_recon_loss(c_out_recon, denorm(c_face))

                optimizer.zero_grad()
                crecon_loss.backward()
                optimizer.step()

                celeba_tloss += crecon_loss.item()

            if data is None and c_data is None:
                break
            
        print('Epoch: {} - Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}, CelebA loss'.format(epoch, tloss, \
                    nloss, aloss, shloss, rloss, celeba_tloss))
        if epoch % 1 == 0:
            print('Training set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}, CelebA Loss: {}'.format(tloss / syn_train_len, \
                    nloss / syn_train_len, aloss / syn_train_len, shloss / syn_train_len, rloss / syn_train_len, celeba_tloss / celeba_train_len))
            # Log training info
            wandb.log({'Train Total loss': tloss/syn_train_len, 'Train Albedo loss': aloss/syn_train_len, 'Train Normal loss': nloss/syn_train_len, \
                        'Train SH loss': shloss/syn_train_len, 'Train Recon loss': rloss/syn_train_len, 'Train CelebA loss:': celeba_tloss/celeba_train_len}, step=epoch)
            
            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' +  'train_' + str(epoch)
            wandb_log_images(wandb, predicted_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo', path=file_name + '_gt_albedo.png')

            # Log CelebA image
            file_name = out_celeba_images_dir + 'train/' +  'train_' + str(epoch)
            wandb_log_images(wandb, c_predicted_normal, c_mask, 'Train CelebA Predicted Normal', epoch, 'Train CelebA Predicted Normal', path=file_name + '_c_predicted_normal.png')
            wandb_log_images(wandb, c_predicted_albedo, c_mask, 'Train CelebA Predicted Albedo', epoch, 'Train CelebA Predicted Albedo', path=file_name + '_c_predicted_albedo.png')
            wandb_log_images(wandb, c_out_shading, c_mask, 'Train CelebA Predicted Shading', epoch, 'Train CelebA Predicted Shading', path=file_name + '_c_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, c_out_recon, c_mask, 'Train CelebA Recon', epoch, 'Train CelebA Recon', path=file_name + '_c_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, c_face, c_mask, 'Train CelebA Ground Truth', epoch, 'Train CelebA Ground Truth', path=file_name + '_c_gt_face.png')

            v_total, v_normal, v_albedo, v_sh, v_recon = predict_sfsnet(sfs_net_model, syn_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                         out_folder=out_syn_images_dir+'/val/', wandb=wandb)

            print('Synthetic Val set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(v_total,
                    v_normal, v_albedo, v_sh, v_recon))
            
            v_total = predict_celeba(sfs_net_model, celeba_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                        out_folder=out_celeba_images_dir+'/val/', wandb=wandb)
            print('CelebA Val set results: Total Loss: {}'.format(v_total))

            
            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
        if epoch % 5 == 0:
            t_total, t_normal, t_albedo, t_sh, t_recon = predict_sfsnet(sfs_net_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_syn_images_dir + '/test/', wandb=wandb)

            print('Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}\n'.format(t_total,
                                                                                                    t_normal, t_albedo, t_sh, t_recon))
