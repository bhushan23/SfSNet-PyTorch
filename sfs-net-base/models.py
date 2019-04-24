import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import denorm

def get_shading(N, L):
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
    
    if torch.cuda.is_available():
        Y1 = Y1.cuda()
        shading = shading.cuda()

    for j in range(c):
        l = sh[j]
        # Scale to 'h x w' dim
        l = l.repeat(1, h*w).view(b, h, w, 9)
        # Convert l into 'batch size', 'Index SH', 'h', 'w'
        l = l.permute([0, 3, 1, 2])
        # Generate shading
        shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                            Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                            Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

    return shading


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
        
        if torch.cuda.is_available():
            Y1 = Y1.cuda()
            shading = shading.cuda()

        for j in range(c):
            l = sh[j]
            # Scale to 'h x w' dim
            l = l.repeat(1, h*w).view(b, h, w, 9)
            # Convert l into 'batch size', 'Index SH', 'h', 'w'
            l = l.permute([0, 3, 1, 2])
            # Generate shading
            shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                                Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                                Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

        return shading


# Base methods for creating convnet
def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )

# SfSNet Models
class ResNetBlock(nn.Module):
    """ Basic building block of ResNet to be used for Normal and Albedo Residual Blocks
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.res = nn.Sequential(
        	nn.BatchNorm2d(in_planes),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(in_planes, in_planes, 3, stride=1, padding=1),
        	nn.BatchNorm2d(in_planes),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(in_planes, out_planes, 3, stride=1, padding=1)
        	)

    def forward(self, x):
        residual = x
        out = self.res(x)
        out += residual

        return out
class baseFeaturesExtractions(nn.Module):
    """ Base Feature extraction
    """
    def __init__(self):
        super(baseFeaturesExtractions, self).__init__()
        self.conv1 = get_conv(3, 64, kernel_size=7, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

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
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1    = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2    = get_conv(128, 64, kernel_size=3, padding=1)
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
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1    = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2    = get_conv(128, 64, kernel_size=3, padding=1)
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
        self.conv1 = get_conv(384, 128, kernel_size=1, stride=1)
        self.pool  = nn.AvgPool2d(64, stride=1,padding=0) 
        self.fc    = nn.Linear(128, 27)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        # reshape to batch_size x 128
        out = out.view(-1, 128)
        out = self.fc(out)
        return out

def reconstruct_image(shading, albedo):
    return shading * albedo
        
class SfsNetPipeline(nn.Module):
    """ SfSNet Pipeline
    """
    def __init__(self):
        super(SfsNetPipeline, self).__init__()

        self.conv_model            = baseFeaturesExtractions()
        self.normal_residual_model = NormalResidualBlock()
        self.normal_gen_model      = NormalGenerationNet()
        self.albedo_residual_model = AlbedoResidualBlock()
        self.albedo_gen_model      = AlbedoGenerationNet()
        self.light_estimator_model = LightEstimator()

    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh)
        recon   = reconstruct_image(shading, albedo)
        return recon

    def forward(self, face):
        # Following is training pipeline
        # 1. Pass Image from Conv Model to extract features
        out_features = self.conv_model(face)

        # 2 a. Pass Conv features through Normal Residual
        out_normal_features = self.normal_residual_model(out_features)
        # 2 b. Pass Conv features through Albedo Residual
        out_albedo_features = self.albedo_residual_model(out_features)
        
        # 3 a. Generate Normal
        predicted_normal = self.normal_gen_model(out_normal_features)
        # 3 b. Generate Albedo
        predicted_albedo = self.albedo_gen_model(out_albedo_features)
        # 3 c. Estimate lighting
        # First, concat conv, normal and albedo features over channels dimension
        all_features = torch.cat((out_features, out_normal_features, out_albedo_features), dim=1)
        # Predict SH
        predicted_sh = self.light_estimator_model(all_features)

        # 4. Generate shading
        out_shading = get_shading(predicted_normal, predicted_sh)

        # 5. Reconstruction of image
        out_recon = reconstruct_image(out_shading, predicted_albedo)

        return predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon


# Following method loads author provided model weights
# Refer to model_loading_synchronization to getf following mapping
# Following mapping is auto-generated using script
def load_model_from_pretrained(src_model, dst_model):
    dst_model['conv_model.conv1.0.weight'] = src_model['conv1.conv.0.weight']
    dst_model['conv_model.conv1.0.bias'] = src_model['conv1.conv.0.bias']
    dst_model['conv_model.conv1.1.weight'] = src_model['conv1.conv.1.weight']
    dst_model['conv_model.conv1.1.bias'] = src_model['conv1.conv.1.bias']
    dst_model['conv_model.conv1.1.running_mean'] = src_model['conv1.conv.1.running_mean']
    dst_model['conv_model.conv1.1.running_var'] = src_model['conv1.conv.1.running_var']
    dst_model['conv_model.conv2.0.weight'] = src_model['conv2.conv.0.weight']
    dst_model['conv_model.conv2.0.bias'] = src_model['conv2.conv.0.bias']
    dst_model['conv_model.conv2.1.weight'] = src_model['conv2.conv.1.weight']
    dst_model['conv_model.conv2.1.bias'] = src_model['conv2.conv.1.bias']
    dst_model['conv_model.conv2.1.running_mean'] = src_model['conv2.conv.1.running_mean']
    dst_model['conv_model.conv2.1.running_var'] = src_model['conv2.conv.1.running_var']
    dst_model['conv_model.conv3.weight'] = src_model['conv3.weight']
    dst_model['conv_model.conv3.bias'] = src_model['conv3.bias']
    dst_model['normal_residual_model.block1.res.0.weight'] = src_model['nres1.res.0.weight']
    dst_model['normal_residual_model.block1.res.0.bias'] = src_model['nres1.res.0.bias']
    dst_model['normal_residual_model.block1.res.0.running_mean'] = src_model['nres1.res.0.running_mean']
    dst_model['normal_residual_model.block1.res.0.running_var'] = src_model['nres1.res.0.running_var']
    dst_model['normal_residual_model.block1.res.2.weight'] = src_model['nres1.res.2.weight']
    dst_model['normal_residual_model.block1.res.2.bias'] = src_model['nres1.res.2.bias']
    dst_model['normal_residual_model.block1.res.3.weight'] = src_model['nres1.res.3.weight']
    dst_model['normal_residual_model.block1.res.3.bias'] = src_model['nres1.res.3.bias']
    dst_model['normal_residual_model.block1.res.3.running_mean'] = src_model['nres1.res.3.running_mean']
    dst_model['normal_residual_model.block1.res.3.running_var'] = src_model['nres1.res.3.running_var']
    dst_model['normal_residual_model.block1.res.5.weight'] = src_model['nres1.res.5.weight']
    dst_model['normal_residual_model.block1.res.5.bias'] = src_model['nres1.res.5.bias']
    dst_model['normal_residual_model.block2.res.0.weight'] = src_model['nres2.res.0.weight']
    dst_model['normal_residual_model.block2.res.0.bias'] = src_model['nres2.res.0.bias']
    dst_model['normal_residual_model.block2.res.0.running_mean'] = src_model['nres2.res.0.running_mean']
    dst_model['normal_residual_model.block2.res.0.running_var'] = src_model['nres2.res.0.running_var']
    dst_model['normal_residual_model.block2.res.2.weight'] = src_model['nres2.res.2.weight']
    dst_model['normal_residual_model.block2.res.2.bias'] = src_model['nres2.res.2.bias']
    dst_model['normal_residual_model.block2.res.3.weight'] = src_model['nres2.res.3.weight']
    dst_model['normal_residual_model.block2.res.3.bias'] = src_model['nres2.res.3.bias']
    dst_model['normal_residual_model.block2.res.3.running_mean'] = src_model['nres2.res.3.running_mean']
    dst_model['normal_residual_model.block2.res.3.running_var'] = src_model['nres2.res.3.running_var']
    dst_model['normal_residual_model.block2.res.5.weight'] = src_model['nres2.res.5.weight']
    dst_model['normal_residual_model.block2.res.5.bias'] = src_model['nres2.res.5.bias']
    dst_model['normal_residual_model.block3.res.0.weight'] = src_model['nres3.res.0.weight']
    dst_model['normal_residual_model.block3.res.0.bias'] = src_model['nres3.res.0.bias']
    dst_model['normal_residual_model.block3.res.0.running_mean'] = src_model['nres3.res.0.running_mean']
    dst_model['normal_residual_model.block3.res.0.running_var'] = src_model['nres3.res.0.running_var']
    dst_model['normal_residual_model.block3.res.2.weight'] = src_model['nres3.res.2.weight']
    dst_model['normal_residual_model.block3.res.2.bias'] = src_model['nres3.res.2.bias']
    dst_model['normal_residual_model.block3.res.3.weight'] = src_model['nres3.res.3.weight']
    dst_model['normal_residual_model.block3.res.3.bias'] = src_model['nres3.res.3.bias']
    dst_model['normal_residual_model.block3.res.3.running_mean'] = src_model['nres3.res.3.running_mean']
    dst_model['normal_residual_model.block3.res.3.running_var'] = src_model['nres3.res.3.running_var']
    dst_model['normal_residual_model.block3.res.5.weight'] = src_model['nres3.res.5.weight']
    dst_model['normal_residual_model.block3.res.5.bias'] = src_model['nres3.res.5.bias']
    dst_model['normal_residual_model.block4.res.0.weight'] = src_model['nres4.res.0.weight']
    dst_model['normal_residual_model.block4.res.0.bias'] = src_model['nres4.res.0.bias']
    dst_model['normal_residual_model.block4.res.0.running_mean'] = src_model['nres4.res.0.running_mean']
    dst_model['normal_residual_model.block4.res.0.running_var'] = src_model['nres4.res.0.running_var']
    dst_model['normal_residual_model.block4.res.2.weight'] = src_model['nres4.res.2.weight']
    dst_model['normal_residual_model.block4.res.2.bias'] = src_model['nres4.res.2.bias']
    dst_model['normal_residual_model.block4.res.3.weight'] = src_model['nres4.res.3.weight']
    dst_model['normal_residual_model.block4.res.3.bias'] = src_model['nres4.res.3.bias']
    dst_model['normal_residual_model.block4.res.3.running_mean'] = src_model['nres4.res.3.running_mean']
    dst_model['normal_residual_model.block4.res.3.running_var'] = src_model['nres4.res.3.running_var']
    dst_model['normal_residual_model.block4.res.5.weight'] = src_model['nres4.res.5.weight']
    dst_model['normal_residual_model.block4.res.5.bias'] = src_model['nres4.res.5.bias']
    dst_model['normal_residual_model.block5.res.0.weight'] = src_model['nres5.res.0.weight']
    dst_model['normal_residual_model.block5.res.0.bias'] = src_model['nres5.res.0.bias']
    dst_model['normal_residual_model.block5.res.0.running_mean'] = src_model['nres5.res.0.running_mean']
    dst_model['normal_residual_model.block5.res.0.running_var'] = src_model['nres5.res.0.running_var']
    dst_model['normal_residual_model.block5.res.2.weight'] = src_model['nres5.res.2.weight']
    dst_model['normal_residual_model.block5.res.2.bias'] = src_model['nres5.res.2.bias']
    dst_model['normal_residual_model.block5.res.3.weight'] = src_model['nres5.res.3.weight']
    dst_model['normal_residual_model.block5.res.3.bias'] = src_model['nres5.res.3.bias']
    dst_model['normal_residual_model.block5.res.3.running_mean'] = src_model['nres5.res.3.running_mean']
    dst_model['normal_residual_model.block5.res.3.running_var'] = src_model['nres5.res.3.running_var']
    dst_model['normal_residual_model.block5.res.5.weight'] = src_model['nres5.res.5.weight']
    dst_model['normal_residual_model.block5.res.5.bias'] = src_model['nres5.res.5.bias']
    dst_model['normal_residual_model.bn1.weight'] = src_model['nreso.0.weight']
    dst_model['normal_residual_model.bn1.bias'] = src_model['nreso.0.bias']
    dst_model['normal_residual_model.bn1.running_mean'] = src_model['nreso.0.running_mean']
    dst_model['normal_residual_model.bn1.running_var'] = src_model['nreso.0.running_var']
    dst_model['normal_gen_model.conv1.0.weight'] = src_model['nconv1.conv.0.weight']
    dst_model['normal_gen_model.conv1.0.bias'] = src_model['nconv1.conv.0.bias']
    dst_model['normal_gen_model.conv1.1.weight'] = src_model['nconv1.conv.1.weight']
    dst_model['normal_gen_model.conv1.1.bias'] = src_model['nconv1.conv.1.bias']
    dst_model['normal_gen_model.conv1.1.running_mean'] = src_model['nconv1.conv.1.running_mean']
    dst_model['normal_gen_model.conv1.1.running_var'] = src_model['nconv1.conv.1.running_var']
    dst_model['normal_gen_model.conv2.0.weight'] = src_model['nconv2.conv.0.weight']
    dst_model['normal_gen_model.conv2.0.bias'] = src_model['nconv2.conv.0.bias']
    dst_model['normal_gen_model.conv2.1.weight'] = src_model['nconv2.conv.1.weight']
    dst_model['normal_gen_model.conv2.1.bias'] = src_model['nconv2.conv.1.bias']
    dst_model['normal_gen_model.conv2.1.running_mean'] = src_model['nconv2.conv.1.running_mean']
    dst_model['normal_gen_model.conv2.1.running_var'] = src_model['nconv2.conv.1.running_var']
    dst_model['normal_gen_model.conv3.weight'] = src_model['nout.weight']
    dst_model['normal_gen_model.conv3.bias'] = src_model['nout.bias']
    dst_model['albedo_residual_model.block1.res.0.weight'] = src_model['ares1.res.0.weight']
    dst_model['albedo_residual_model.block1.res.0.bias'] = src_model['ares1.res.0.bias']
    dst_model['albedo_residual_model.block1.res.0.running_mean'] = src_model['ares1.res.0.running_mean']
    dst_model['albedo_residual_model.block1.res.0.running_var'] = src_model['ares1.res.0.running_var']
    dst_model['albedo_residual_model.block1.res.2.weight'] = src_model['ares1.res.2.weight']
    dst_model['albedo_residual_model.block1.res.2.bias'] = src_model['ares1.res.2.bias']
    dst_model['albedo_residual_model.block1.res.3.weight'] = src_model['ares1.res.3.weight']
    dst_model['albedo_residual_model.block1.res.3.bias'] = src_model['ares1.res.3.bias']
    dst_model['albedo_residual_model.block1.res.3.running_mean'] = src_model['ares1.res.3.running_mean']
    dst_model['albedo_residual_model.block1.res.3.running_var'] = src_model['ares1.res.3.running_var']
    dst_model['albedo_residual_model.block1.res.5.weight'] = src_model['ares1.res.5.weight']
    dst_model['albedo_residual_model.block1.res.5.bias'] = src_model['ares1.res.5.bias']
    dst_model['albedo_residual_model.block2.res.0.weight'] = src_model['ares2.res.0.weight']
    dst_model['albedo_residual_model.block2.res.0.bias'] = src_model['ares2.res.0.bias']
    dst_model['albedo_residual_model.block2.res.0.running_mean'] = src_model['ares2.res.0.running_mean']
    dst_model['albedo_residual_model.block2.res.0.running_var'] = src_model['ares2.res.0.running_var']
    dst_model['albedo_residual_model.block2.res.2.weight'] = src_model['ares2.res.2.weight']
    dst_model['albedo_residual_model.block2.res.2.bias'] = src_model['ares2.res.2.bias']
    dst_model['albedo_residual_model.block2.res.3.weight'] = src_model['ares2.res.3.weight']
    dst_model['albedo_residual_model.block2.res.3.bias'] = src_model['ares2.res.3.bias']
    dst_model['albedo_residual_model.block2.res.3.running_mean'] = src_model['ares2.res.3.running_mean']
    dst_model['albedo_residual_model.block2.res.3.running_var'] = src_model['ares2.res.3.running_var']
    dst_model['albedo_residual_model.block2.res.5.weight'] = src_model['ares2.res.5.weight']
    dst_model['albedo_residual_model.block2.res.5.bias'] = src_model['ares2.res.5.bias']
    dst_model['albedo_residual_model.block3.res.0.weight'] = src_model['ares3.res.0.weight']
    dst_model['albedo_residual_model.block3.res.0.bias'] = src_model['ares3.res.0.bias']
    dst_model['albedo_residual_model.block3.res.0.running_mean'] = src_model['ares3.res.0.running_mean']
    dst_model['albedo_residual_model.block3.res.0.running_var'] = src_model['ares3.res.0.running_var']
    dst_model['albedo_residual_model.block3.res.2.weight'] = src_model['ares3.res.2.weight']
    dst_model['albedo_residual_model.block3.res.2.bias'] = src_model['ares3.res.2.bias']
    dst_model['albedo_residual_model.block3.res.3.weight'] = src_model['ares3.res.3.weight']
    dst_model['albedo_residual_model.block3.res.3.bias'] = src_model['ares3.res.3.bias']
    dst_model['albedo_residual_model.block3.res.3.running_mean'] = src_model['ares3.res.3.running_mean']
    dst_model['albedo_residual_model.block3.res.3.running_var'] = src_model['ares3.res.3.running_var']
    dst_model['albedo_residual_model.block3.res.5.weight'] = src_model['ares3.res.5.weight']
    dst_model['albedo_residual_model.block3.res.5.bias'] = src_model['ares3.res.5.bias']
    dst_model['albedo_residual_model.block4.res.0.weight'] = src_model['ares4.res.0.weight']
    dst_model['albedo_residual_model.block4.res.0.bias'] = src_model['ares4.res.0.bias']
    dst_model['albedo_residual_model.block4.res.0.running_mean'] = src_model['ares4.res.0.running_mean']
    dst_model['albedo_residual_model.block4.res.0.running_var'] = src_model['ares4.res.0.running_var']
    dst_model['albedo_residual_model.block4.res.2.weight'] = src_model['ares4.res.2.weight']
    dst_model['albedo_residual_model.block4.res.2.bias'] = src_model['ares4.res.2.bias']
    dst_model['albedo_residual_model.block4.res.3.weight'] = src_model['ares4.res.3.weight']
    dst_model['albedo_residual_model.block4.res.3.bias'] = src_model['ares4.res.3.bias']
    dst_model['albedo_residual_model.block4.res.3.running_mean'] = src_model['ares4.res.3.running_mean']
    dst_model['albedo_residual_model.block4.res.3.running_var'] = src_model['ares4.res.3.running_var']
    dst_model['albedo_residual_model.block4.res.5.weight'] = src_model['ares4.res.5.weight']
    dst_model['albedo_residual_model.block4.res.5.bias'] = src_model['ares4.res.5.bias']
    dst_model['albedo_residual_model.block5.res.0.weight'] = src_model['ares5.res.0.weight']
    dst_model['albedo_residual_model.block5.res.0.bias'] = src_model['ares5.res.0.bias']
    dst_model['albedo_residual_model.block5.res.0.running_mean'] = src_model['ares5.res.0.running_mean']
    dst_model['albedo_residual_model.block5.res.0.running_var'] = src_model['ares5.res.0.running_var']
    dst_model['albedo_residual_model.block5.res.2.weight'] = src_model['ares5.res.2.weight']
    dst_model['albedo_residual_model.block5.res.2.bias'] = src_model['ares5.res.2.bias']
    dst_model['albedo_residual_model.block5.res.3.weight'] = src_model['ares5.res.3.weight']
    dst_model['albedo_residual_model.block5.res.3.bias'] = src_model['ares5.res.3.bias']
    dst_model['albedo_residual_model.block5.res.3.running_mean'] = src_model['ares5.res.3.running_mean']
    dst_model['albedo_residual_model.block5.res.3.running_var'] = src_model['ares5.res.3.running_var']
    dst_model['albedo_residual_model.block5.res.5.weight'] = src_model['ares5.res.5.weight']
    dst_model['albedo_residual_model.block5.res.5.bias'] = src_model['ares5.res.5.bias']
    dst_model['albedo_residual_model.bn1.weight'] = src_model['areso.0.weight']
    dst_model['albedo_residual_model.bn1.bias'] = src_model['areso.0.bias']
    dst_model['albedo_residual_model.bn1.running_mean'] = src_model['areso.0.running_mean']
    dst_model['albedo_residual_model.bn1.running_var'] = src_model['areso.0.running_var']
    dst_model['albedo_gen_model.conv1.0.weight'] = src_model['aconv1.conv.0.weight']
    dst_model['albedo_gen_model.conv1.0.bias'] = src_model['aconv1.conv.0.bias']
    dst_model['albedo_gen_model.conv1.1.weight'] = src_model['aconv1.conv.1.weight']
    dst_model['albedo_gen_model.conv1.1.bias'] = src_model['aconv1.conv.1.bias']
    dst_model['albedo_gen_model.conv1.1.running_mean'] = src_model['aconv1.conv.1.running_mean']
    dst_model['albedo_gen_model.conv1.1.running_var'] = src_model['aconv1.conv.1.running_var']
    dst_model['albedo_gen_model.conv2.0.weight'] = src_model['aconv2.conv.0.weight']
    dst_model['albedo_gen_model.conv2.0.bias'] = src_model['aconv2.conv.0.bias']
    dst_model['albedo_gen_model.conv2.1.weight'] = src_model['aconv2.conv.1.weight']
    dst_model['albedo_gen_model.conv2.1.bias'] = src_model['aconv2.conv.1.bias']
    dst_model['albedo_gen_model.conv2.1.running_mean'] = src_model['aconv2.conv.1.running_mean']
    dst_model['albedo_gen_model.conv2.1.running_var'] = src_model['aconv2.conv.1.running_var']
    dst_model['albedo_gen_model.conv3.weight'] = src_model['aout.weight']
    dst_model['albedo_gen_model.conv3.bias'] = src_model['aout.bias']
    dst_model['light_estimator_model.conv1.0.weight'] = src_model['lconv.conv.0.weight']
    dst_model['light_estimator_model.conv1.0.bias'] = src_model['lconv.conv.0.bias']
    dst_model['light_estimator_model.conv1.1.weight'] = src_model['lconv.conv.1.weight']
    dst_model['light_estimator_model.conv1.1.bias'] = src_model['lconv.conv.1.bias']
    dst_model['light_estimator_model.conv1.1.running_mean'] = src_model['lconv.conv.1.running_mean']
    dst_model['light_estimator_model.conv1.1.running_var'] = src_model['lconv.conv.1.running_var']
    dst_model['light_estimator_model.fc.weight'] = src_model['lout.weight']
    dst_model['light_estimator_model.fc.bias'] = src_model['lout.bias']
    return dst_model