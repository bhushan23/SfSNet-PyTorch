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

def get_dataset(dir, validation_split=0):
    albedo  = []
    sh      = []
    mask    = []
    normal  = []
    face    = []
    depth   = []

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

    # Debugging print out images considered
    # with open('albedo.txt', 'w') as f:
    #     for item in albedo:
    #         f.write("%s\n" % item)    

    # with open('face.txt', 'w') as f:
    #     for item in face:
    #         f.write("%s\n" % item)    

    # with open('normal.txt', 'w') as f:
    #     for item in normal:
    #         f.write("%s\n" % item)    

    # with open('mask.txt', 'w') as f:
    #     for item in mask:
    #         f.write("%s\n" % item)    
    print(len(albedo) , len(face) , len(normal) , len(depth) , len(mask) , len(sh))
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

