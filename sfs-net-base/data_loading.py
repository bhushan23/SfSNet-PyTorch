import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import glob
import cv2
from random import randint
import os
from skimage import io
from PIL import Image
import pandas as pd

from utils import save_image
import numpy as np
IMAGE_SIZE = 128

def generate_sfsnet_data_csv(dir, save_location):
    albedo = set()
    normal = set()
    depth  = set()
    mask   = set()
    face   = set()
    sh     = set()

    name_to_set = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh}
    
    for k, v in name_to_set.items():
        regex_str = '*/*_' + k + '_*'
        for img in sorted(glob.glob(dir + regex_str)):
            timg = img.split('/')
            folder_id = timg[-2]
            name      = timg[-1].split('.')[0]
            name      = name.split('_')
            assert(len(name) == 4)
            name      = folder_id + '_' + name[0] + '_' + name[2] + '_' + name[3]
            v.add(name)

    final_images = set.intersection(albedo, normal, depth, mask, face, sh)    

    albedo = []
    normal = []
    depth  = []
    mask   = []
    face   = []
    sh     = []
    name   = []

    name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh, 'name' : name}

    for img in final_images:
        split = img.split('_')
        for k, v in name_to_list.items():
            ext = '.png'
            if k == 'light':
                ext = '.txt'

            if k == 'name':
                filename = split[0] + '_' + split[1] + '_' + k + '_' + '_'.join(split[2:])
            else:
                file_name = split[0] + '/' + split[1] + '_' + k + '_' + '_'.join(split[2:]) + ext
            v.append(file_name)

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)

def generate_celeba_synthesize_data_csv(dir, save_location):
    albedo = []
    normal = []
    depth  = []
    mask   = []
    face   = []
    sh     = []
    name   = []

    name_to_set = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh, 'name' : name}
    
    for img in sorted(glob.glob(dir + '*_albedo*')):
        albedo.append(img)
    
    for img in sorted(glob.glob(dir + '*_normal*')):
        normal.append(img)

    for img in sorted(glob.glob(dir + '*_face*')):
        face.append(img)
        mask.append('None')
        depth.append('None')
        iname = img.split('/')[-1].split('.')[0]
        name.append(iname)

    for l in sorted(glob.glob(dir + '*_light*')):
        sh.append(l)

    name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh, 'name' : name}

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)

def generate_celeba_data_csv(dir, save_location):
    face = []
    name = []

    for img in sorted(glob.glob(dir + '*/all/*.jpg')):
        face.append(img)
        iname = img.split('/')[-1].split('.')[0]
        name.append(iname)

    face_to_list = {'face': face, 'name':name}
    df = pd.DataFrame(data=face_to_list)
    df.to_csv(save_location)

def get_sfsnet_dataset(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0):
    albedo  = []
    sh      = []
    mask    = []
    normal  = []
    face    = []
    depth   = []

    if read_from_csv is None:
        for img in sorted(glob.glob(syn_dir + '*/*_albedo_*')):
            albedo.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_face_*')):
            face.append(img)    

        for img in sorted(glob.glob(syn_dir + '*/*_normal_*')):
            normal.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_depth_*')):
            depth.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_mask_*')):
            mask.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_light_*.txt')):
            sh.append(img)
    else:
        df = pd.read_csv(read_from_csv)
        df = df[:read_first]
        albedo = list(df['albedo'])
        face   = list(df['face'])
        normal = list(df['normal'])
        depth  = list(df['depth'])
        mask   = list(df['mask'])
        sh     = list(df['light'])

        name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh}

        for _, v in name_to_list.items():
            v[:] = [syn_dir + el for el in v]

        # Merge Synthesized Celeba dataset for Psedo-Supervised training
        if read_celeba_csv is not None:
            df = pd.read_csv(read_celeba_csv)
            df = df[:read_first]
            albedo += list(df['albedo'])
            face   += list(df['face'])
            normal += list(df['normal'])
            depth  += list(df['depth'])
            mask   += list(df['mask'])
            sh     += list(df['light'])

    assert(len(albedo) == len(face) == len(normal) == len(depth) == len(mask) == len(sh))
    dataset_size = len(albedo)
    validation_count = int (validation_split * dataset_size / 100)
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
            ])
    
    full_dataset = SfSNetDataset(albedo, face, normal, mask, sh, transform)  
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def get_celeba_dataset(dir=None, read_from_csv=None, read_first=None, validation_split=0):
    face    = []

    if read_from_csv is None:
        for img in sorted(glob.glob(dir + '*/*_face_*')):
            face.append(img)    
    else:
        df = pd.read_csv(read_from_csv)
        df = df[:read_first]
        face   = list(df['face'])

    dataset_size = len(face)
    validation_count = int (validation_split * dataset_size / 100)
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
                ])
    
    full_dataset = CelebADataset(face, transform)  
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def generate_celeba_synthesize(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):
 
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
        
        # save predictions in log folder
        file_name = out_folder + str(train_epoch_num) + '_' + str(bix)
        # log images
        save_image(predicted_normal, path = file_name+'_normal.png')
        save_image(predicted_albedo, path = file_name+'_albedo.png')
        save_image(predicted_shading, path = file_name+'_shading.png')
        save_image(predicted_face, path = file_name+'_recon.png')
        save_image(face, path = file_name+'_face.png')
        np.savetxt(file_name+'_light.txt', predicted_sh.cpu().detach().numpy(), delimiter='\t')
        
        # Loss computation
        # Reconstruction loss
        total_loss  = recon_loss(predicted_face, face)

        # Logging for display and debugging purposes
        tloss += total_loss.item()
    
    len_dl = len(dl)

    f = open(out_folder + 'readme.txt', 'w')
    f.write('Average Reconstruction Loss: ' + str(tloss / len_dl))
    f.close()

    # return average loss over dataset
    return tloss / len_dl

class SfSNetDataset(Dataset):
    def __init__(self, albedo, face, normal, mask, sh, transform = None):
        self.albedo = albedo
        self.face   = face
        self.normal = normal
        self.mask   = mask
        self.sh     = sh
        self.transform = transform
        self.dataset_len = len(self.albedo)
        self.mask_transform = transforms.Compose([
                              transforms.Resize(IMAGE_SIZE),
                              transforms.ToTensor(),
                            ])
        self.normal_transform = transforms.Compose([
                              transforms.Resize(IMAGE_SIZE)
                            ])

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.albedo[index]))
        face   = self.transform(Image.open(self.face[index]))
        # normal = io.imread(self.face[index]))
        normal = self.normal_transform(Image.open(self.normal[index]))
        normal = torch.tensor(np.asarray(normal)).permute([2, 0, 1])
        normal = normal.type(torch.float)
        normal = (normal - 128) / 128
        if self.mask[index] == 'None':
            # Load dummy 1 mask for CelebA
            # To ensure consistency if mask is used
            mask = torch.ones(3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            mask   = self.mask_transform(Image.open(self.mask[index]))
        pd_sh  = pd.read_csv(self.sh[index], sep='\t', header = None)
        sh     = torch.tensor(pd_sh.values).type(torch.float).reshape(-1)
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len

class CelebADataset(Dataset):
    def __init__(self, face, transform = None):
        self.face   = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.mask_transform = transforms.Compose([
                              transforms.Resize(IMAGE_SIZE),
                              transforms.ToTensor()
                            ])

    def __getitem__(self, index):
        face   = self.transform(Image.open(self.face[index]))
        return face

    def __len__(self):
        return self.dataset_len


