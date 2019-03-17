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

IMAGE_SIZE = 64

def get_dataset(dir, validation_split=0):
    albedo  = []
    sh      = []
    mask    = []
    normal  = []
    face    = []
    depth   = []

    for img in sorted(glob.glob(dir + '*/*_albedo_*'), key=os.path.getmtime):
        albedo.append(img)

    for img in sorted(glob.glob(dir + '*/*_face_*'), key=os.path.getmtime):
        face.append(img)    

    for img in sorted(glob.glob(dir + '*/*_normal_*'), key=os.path.getmtime):
        normal.append(img)

    for img in sorted(glob.glob(dir + '*/*_depth_*'), key=os.path.getmtime):
        depth.append(img)

    for img in sorted(glob.glob(dir + '*/*_mask_*'), key=os.path.getmtime):
        mask.append(img)

    for img in sorted(glob.glob(dir + '*/*_light_*.txt'), key=os.path.getmtime):
        sh.append(img)

    assert(len(albedo) == len(face) == len(normal) == len(depth) == len(mask) == len(sh))
    dataset_size = len(albedo)
    validation_count = int (validation_split * dataset_size / 100)
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    
    full_dataset = SfSNetDataset(albedo, face, normal, mask, sh, transform)  

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

    def __getitem__(self, index):
        print(self.albedo[index])
        albedo = self.transform(Image.open(self.albedo[index]))
        face   = self.transform(Image.open(self.face[index]))
        normal = self.transform(Image.open(self.normal[index]))
        mask   = self.transform(Image.open(self.mask[index]))
        pd_sh  = pd.read_csv(self.sh[index], sep='\t', header = None)
        sh     = torch.tensor(pd_sh.values)

        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len

