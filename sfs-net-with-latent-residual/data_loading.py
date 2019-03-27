import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import glob
import cv2
from random import randint
import os
from skimage import io
from PIL import Image
import pandas as pd

IMAGE_SIZE = 128

def generate_data_csv(dir, save_location):
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

    name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh}

    for img in final_images:
        split = img.split('_')
        for k, v in name_to_list.items():
            ext = '.png'
            if k == 'light':
                ext = '.txt'
            file_name = split[0] + '/' + split[1] + '_' + k + '_' + '_'.join(split[2:]) + ext
            v.append(file_name)

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)

def get_dataset(dir, read_from_csv=None, validation_split=0):
    albedo  = []
    sh      = []
    mask    = []
    normal  = []
    face    = []
    depth   = []

    if read_from_csv is None:
        for img in sorted(glob.glob(dir + '*/*_albedo_*')):
            albedo.append(img)

        for img in sorted(glob.glob(dir + '*/*_face_*')):
            face.append(img)    

        for img in sorted(glob.glob(dir + '*/*_normal_*')):
            normal.append(img)

        for img in sorted(glob.glob(dir + '*/*_depth_*')):
            depth.append(img)

        for img in sorted(glob.glob(dir + '*/*_mask_*')):
            mask.append(img)

        for img in sorted(glob.glob(dir + '*/*_light_*.txt')):
            sh.append(img)
    else:
        df = pd.read_csv(read_from_csv)
        albedo = list(df['albedo'])
        face   = list(df['face'])
        normal = list(df['normal'])
        depth  = list(df['depth'])
        mask   = list(df['mask'])
        sh     = list(df['light'])

        name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh}

        for _, v in name_to_list.items():
            v[:] = [dir + el for el in v]

    assert(len(albedo) == len(face) == len(normal) == len(depth) == len(mask) == len(sh))
    dataset_size = len(albedo)
    validation_count = int (validation_split * dataset_size / 100)
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    
    full_dataset = SfSNetDataset(albedo, face, normal, mask, sh, transform)  
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

# dataset_path='/nfs/bigdisk/bsonawane/sfsnet_data/'
# dataset_path = '../data/'
# generate_data_csv(dataset_path + 'train/', dataset_path + '/train.csv')
# generate_data_csv(dataset_path + 'test/', dataset_path + '/test.csv')

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

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.albedo[index]))
        face   = self.transform(Image.open(self.face[index]))
        normal = self.transform(Image.open(self.normal[index]))
        mask   = self.mask_transform(Image.open(self.mask[index]))
        pd_sh  = pd.read_csv(self.sh[index], sep='\t', header = None)
        sh     = torch.tensor(pd_sh.values).type(torch.float)
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len

