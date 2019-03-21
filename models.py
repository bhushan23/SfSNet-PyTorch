import torch
import torch.nn as nn
import torch.functional as F

class sfsNetShading(nn.Module):
    def __init__(self):
        super(sfsNetShading, self).__init__()
    
    def forward(self, N, L):
        # Following values are computed from equation
        # from SFSNet
        c1 = 0.8862269254527579
        c2 = 1.0233267079464883
        c3 = 0.24770795610037571
        c4 = 0.8580855308097834
        c5 = 0.4290427654048917

        nx = N[:, 0, :, :]
        ny = N[:, 1, :, :]
        nz = N[:, 2, :, :]
        
        b, c, h, w = N.shape
        
        Y1 = c1 * torch.ones(b, h, w)
        Y2 = c2 * nz
        Y3 = c2 * nx
        Y4 = c2 * ny
        Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
        Y6 = c4 * nx * nz
        Y7 = c4 * ny * nz
        Y8 = c5 * (nx * nx - ny * ny)
        Y9 = c4 * nx * ny

        L = L.type(torch.float)
        sh = torch.split(L, 9, dim=1)

        assert(c == len(sh))
        shading = torch.zeros(b, c, h, w)
        for j in range(c):
            l = sh[j]
            shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                                  Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                                  Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

        return shading


# Base methods for creating convnet
def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(dropout),
        nn.Dropout(dropout)
    )

# SfSNet Models
class ResNetBlock(nn.Module):
    """ Basic building block of ResNet to be used for Normal and Albedo Residual Blocks
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class baseFeaturesExtractions(nn.Module):
    """ Base Feature extraction
    """
    def __init__(self):
        super(baseFeaturesExtractions, self).__init__()
        self.conv1 = get_conv(3, 64, kernel_size=7, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, padding=1)
        self.conv3 = get_conv(128, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class NormalResidualBlock(nn.Module):
    """ Net to general Normal from features
    """
    def __init__(self):
        super(NormalResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1    = nn.BatchNorm2d(128)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out

class AlbedoResidualBlock(nn.Module):
    """ Net to general Albedo from features
    """
    def __init__(self):
        super(AlbedoResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1    = nn.BatchNorm2d(128)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out

class NormalGenerationNet(nn.Module):
    """ Generating Normal
    """
    def __init__(self):
        super(NormalGenerationNet, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=None)
        self.conv1    = get_conv(128, 64, kernel_size=1, stride=1)
        self.conv2    = get_conv(64, 64, kernel_size=3, padding=1)
        self.conv3    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class AlbedoGenerationNet(nn.Module):
    """ Generating Albedo
    """
    def __init__(self):
        super(AlbedoGenerationNet, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=None)
        self.conv1    = get_conv(128, 64, kernel_size=1, stride=1)
        self.conv2    = get_conv(64, 64, kernel_size=3, padding=1)
        self.conv3    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class LightEstimator(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv1 = nn.Conv2d(384, 128, kernel_size=1)
        self.pool  = nn.AvgPool2d(64) 
        self.fc    = nn.Linear(128, 27)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.fc(out)
        return out
        
