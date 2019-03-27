import torch
import torch.nn as nn
import numpy as np
import os

from models import *
from utils import *

## TODOS:
## 1. Dump SH in file
## 
## 
## Notes:
## 1. SH is not normalized
## 2. Face is normalized and denormalized - shall we not normalize in the first place?


# Enable WANDB Logging
WANDB_ENABLE = True

def test_celeba(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):
    for bix, data in enumerate(dl):
        face, mask = data
        if use_cuda:
            face   = face.cuda()
            mask   = mask.cuda()

        # Apply Mask on input image
        face = applyMask(face, mask)
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_corrected_shading, predicted_face = sfsnet_net_model(face)

        # save predictions in log folder
        file_name = out_folder + 'test_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
        save_image(predicted_normal, path=file_name + '_predicted_normal.png', mask=mask) 
        save_image(predicted_albedo, path=file_name + '_predicted_albedo.png', mask=mask) 
        save_image(predicted_face, denormalize=False, path=file_name + '_predicted_face.png', mask=mask)
        save_image(predicted_shading, denormalize=False, path=file_name + '_predicted_shading.png', mask = mask)
        save_image(face, path=file_name + '_gt_face.png', mask=mask)

        if WANDB_ENABLE:
            wandb_log_images(wandb, predicted_normal, mask, 'Test Predicted Normal', train_epoch_num, 'Test Predicted Normal')
            wandb_log_images(wandb, predicted_albedo, mask, 'Test Predicted Albedo', train_epoch_num, 'Test Predicted Albedo')
            wandb_log_images(wandb, predicted_shading, mask, 'Test Predicted Shading', train_epoch_num, 'Test Predicted Shading', denormalize=False)
            wandb_log_images(wandb, predicted_face, mask, 'Test Recon', train_epoch_num, 'Test Recon', denormalize=False)
            wandb_log_images(wandb, face, mask, 'Test Ground Truth Face', train_epoch_num, 'Test Ground Truth Face')

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
        face = applyMask(face, mask)
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_corrected_shading, predicted_face = sfs_net_model(face)
        # predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfsnet_pipeline(conv_model, normal_residual_model, albedo_residual_model,
        #                                                                light_estimator_model, normal_gen_model, albedo_gen_model,
        #                                                                shading_model, image_recon_model, face)
        if bix == fix_bix_dump:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
            save_image(predicted_normal, path=file_name + '_predicted_normal.png', mask=mask) 
            save_image(predicted_albedo, path=file_name + '_predicted_albedo.png', mask=mask) 
            save_image(predicted_face, denormalize=False, path=file_name + '_predicted_face.png', mask=mask)
            save_image(face, path=file_name + '_gt_face.png', mask=mask)
            save_image(predicted_shading, denormalize=False, path=file_name + '_predicted_shading.png', mask = mask)
            save_image(predicted_corrected_shading, denormalize=False, path=file_name + '_predicted_corrected_shading.png', mask = mask)
            save_image(normal, path=file_name + '_normal.png', mask=mask) 
            save_image(albedo, path=file_name + '_albedo.png', mask=mask) 
            # log images
            if WANDB_ENABLE:
                wandb_log_images(wandb, predicted_normal, mask, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal')
                wandb_log_images(wandb, predicted_albedo, mask, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo')
                wandb_log_images(wandb, predicted_shading, mask, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', denormalize=False)
                wandb_log_images(wandb, predicted_corrected_shading, mask, suffix+' Predicted Corrected Shading', train_epoch_num, suffix+' Predicted Corrected Shading', denormalize=False)
                wandb_log_images(wandb, predicted_face, mask, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', denormalize=False)
                wandb_log_images(wandb, face, mask, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth')
                wandb_log_images(wandb, normal, mask, suffix+' Ground Truth Normal', train_epoch_num, suffix+' Ground Normal')
                wandb_log_images(wandb, albedo, mask, suffix+' Ground Truth Albedo', train_epoch_num, suffix+' Ground Albedo')

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

def train(sfs_net_model, train_dl, val_dl, test_dl,
          num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):


    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_val_images_dir   = out_images_dir + 'val/'
    out_test_images_dir  = out_images_dir + 'test/'
    out_train_images_dir = out_images_dir + 'train/'

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_val_images_dir))
    os.system('mkdir -p {}'.format(out_test_images_dir))
    os.system('mkdir -p {}'.format(out_train_images_dir))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
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

    train_set_len = len(train_dl)
    val_set_len   = len(val_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        nloss = 0 # Normal loss
        aloss = 0 # Albedo loss
        shloss = 0 # SH loss
        rloss = 0 # Reconstruction loss

        for _, data in enumerate(train_dl):
            albedo, normal, mask, sh, face = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask   = mask.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
            
            # Apply Mask on input image
            face = applyMask(face, mask)

            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_predicted_shading, out_recon = sfs_net_model(face)

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
        if epoch % 1 == 0:
                        
            print('Training set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(tloss / train_set_len, \
                    nloss / train_set_len, aloss / train_set_len, shloss / train_set_len, rloss / train_set_len))
            v_total, v_normal, v_albedo, v_sh, v_recon = predict_sfsnet(sfs_net_model, val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                        out_folder=out_val_images_dir, wandb=wandb)

            print('Val set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(v_total,
                    v_normal, v_albedo, v_sh, v_recon))
            
            file_name = out_train_images_dir +  'train_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
            save_image(predicted_normal, path=file_name + '_predicted_normal.png', mask=mask) 
            save_image(predicted_albedo, path=file_name + '_predicted_albedo.png', mask=mask) 
            save_image(out_recon, denormalize=False, path=file_name + '_predicted_face.png', mask=mask)
            save_image(face, path=file_name + '_gt_face.png', mask=mask)
            save_image(out_shading, denormalize=False, path=file_name + '_predicted_shading.png', mask = mask)
            save_image(out_corrected_shading, denormalize=False, path=file_name + '_predicted_corrected_shading.png', mask = mask)
            save_image(normal, path=file_name + '_normal.png', mask=mask) 
            save_image(albedo, path=file_name + '_albedo.png', mask=mask) 

            # Log training info
            wandb.log({'Train Total loss': tloss/train_set_len, 'Train Albedo loss': aloss/train_set_len, 'Train Normal loss': nloss/train_set_len, \
                        'Train SH loss': shloss/train_set_len, 'Train Recon loss': rloss/train_set_len}, step=epoch)
            
            # Log images in wandb
            if WANDB_ENABLE:
                wandb_log_images(wandb, predicted_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal')
                wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo')
                wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', denormalize=False)
                wandb_log_images(wandb, out_predicted_shading, mask, 'Train Predicted Corrected Shading', epoch, 'Train Predicted Corrected Shading', denormalize=False)
                wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', denormalize=False)
                wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth')
                wandb_log_images(wandb, normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal')
                wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo')

            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
        if epoch % 5 == 0:
            t_total, t_normal, t_albedo, t_sh, t_recon = predict_sfsnet(sfs_net_model, test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_test_images_dir, wandb=wandb)

            print('Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}\n'.format(t_total,
                                                                                                    t_normal, t_albedo, t_sh, t_recon))
