import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import pandas as pd

from utils import *
from models import sfsNetShading

# def var(x):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x)

# Start of log based shading generation method
# Credits: Zhixin Shu for providing following method
class waspShadeRenderer(nn.Module):
    def __init__(self, opt):
        super(waspShadeRenderer, self).__init__()
        self.opt = opt
        self.getHomo = HomogeneousCoord(opt)
        self.getMMatrix = MMatrix(opt)

    def forward(self, light, normals):
        # homogeneous coordinate of the normals
        #normals = var(normals).type(torch.DoubleTensor)
        batchSize = normals.size(0)
        W = normals.size(2)
        H = normals.size(3)
        hNormals = self.getHomo(normals)
        # matrix for light
        mLight = self.getMMatrix(light)
        # get shading from these two: N x 4 , N = batchSize x W x H
        hN_vec = hNormals.view(batchSize, 4, -1).permute(0,2,1).contiguous().view(-1,4)
        # N x 1 x 4
        hN_vec_Left  = hN_vec.unsqueeze(1)
        # N x 4 x 1
        hN_vec_Right = hN_vec.unsqueeze(2)
        # expand the lighting from batchSize x 4 x 4 to N x 4 x 4
        hL = mLight.view(batchSize,16).repeat(1,W*H).view(-1,4,4).type(torch.float)
        shade0 = torch.matmul(hN_vec_Left, hL)
        shade1 = torch.matmul(shade0, hN_vec_Right)
        #shade1 is tensor of size Nx1x1 = batchSize x W x H
        shading = shade1.view(batchSize,W,H).unsqueeze(1)
        return shading

class HomogeneousCoord(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(HomogeneousCoord, self).__init__()
        self.opt = opt
    def forward(self, x):
        y = torch.ones(x.size(0),1,x.size(2),x.size(3))
        z = torch.cat((x,y),1)
        return z

class MMatrix(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(MMatrix, self).__init__()
        self.opt = opt
    
    def forward(self, L):
        # input L:[batchSize,9]
        # output M: [batchSize, 4, 4]
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743152
        c4 = 0.886227
        c5 = 0.247708
        M00 = c1*L[:,8].unsqueeze(1)
        M01 = c1*L[:,4].unsqueeze(1)
        M02 = c1*L[:,7].unsqueeze(1)
        M03 = c2*L[:,3].unsqueeze(1)
        M10 = c1*L[:,4].unsqueeze(1)
        M11 = -c1*L[:,8].unsqueeze(1)
        M12 = c1*L[:,5].unsqueeze(1)
        M13 = c2*L[:,1].unsqueeze(1)
        M20 = c1*L[:,7].unsqueeze(1)
        M21 = c1*L[:,5].unsqueeze(1)
        M22 = c3*L[:,6].unsqueeze(1)
        M23 = c2*L[:,2].unsqueeze(1)
        M30 = c2*L[:,3].unsqueeze(1)
        M31 = c2*L[:,1].unsqueeze(1)
        M32 = c2*L[:,2].unsqueeze(1)
        M33 = c4*L[:,0].unsqueeze(1) - c5*L[:,6].unsqueeze(1)
        M0 = torch.cat((M00,M01,M02,M03),dim=1).unsqueeze(1)
        M1 = torch.cat((M10,M11,M12,M13),dim=1).unsqueeze(1)
        M2 = torch.cat((M20,M21,M22,M23),dim=1).unsqueeze(1)
        M3 = torch.cat((M30,M31,M32,M33),dim=1).unsqueeze(1)
        M = torch.cat((M0,M1,M2,M3),dim=1)
        return M

# End of log based shading generation method

def getShadingFromNormalAndSH(Normal, rSH):
    shader = waspShadeRenderer(None)
    #print('SHader size:', Normal.size())
    rSH  = rSH.view(rSH.shape[0], rSH.shape[2])
    sh1, sh2, sh3 = torch.split(rSH, 9, dim=1)
    out1 = shader(sh1, Normal)
    out2 = shader(sh2, Normal)
    out3 = shader(sh3, Normal)
    outShadingB = torch.cat((out1, out2, out3), 1)
    return outShadingB


def validate_shading_method(train_dl):
    albedo, normal, mask, sh, face = next(iter(train_dl))

    shading = getShadingFromNormalAndSH(normal, sh)
    save_image(albedo,  denormalize=False, mask=mask, path='./results/shading_from_normal/albedo.png')
    save_image(normal,  denormalize=False, mask=mask, path='./results/shading_from_normal/normal.png')
    save_image(shading, denormalize=False, mask=mask, path='./results/shading_from_normal/shading_ours.png')

    recon   = shading * albedo
    save_image(recon, mask=mask, denormalize=False, path='./results/shading_from_normal/recon_ours.png')
    save_image(face, mask=mask, path = './results/shading_from_normal/recon_groundtruth.png')

    recon = applyMask(recon, mask)
    face  = applyMask(face, mask)
    mseLoss = nn.L1Loss()
    print('L1Loss Ours: ', mseLoss(face, recon).item())

    sfsnet_shading_net = sfsNetShading()
    sh = sh.view(sh.shape[0], sh.shape[2])
    sfs_shading = sfsnet_shading_net(normal, sh)
    save_image(sfs_shading, mask=mask, denormalize=False, path='./results/shading_from_normal/shading_sfsnet.png')
    recon   = sfs_shading * albedo
    save_image(recon, mask=mask, denormalize=False, path='./results/shading_from_normal/recon_sfsnet.png')

    recon = applyMask(recon, mask)
    face  = applyMask(face, mask)
    mseLoss = nn.L1Loss()
    print('L1Loss SFSNet: ', mseLoss(face, recon).item())