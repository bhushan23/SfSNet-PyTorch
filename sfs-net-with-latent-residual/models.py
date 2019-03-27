import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import denorm

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
        self.bn1 = nn.BatchNorm2d(out_planes)
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
        # reshape to batch_size x 128
        out = out.view(-1, 128)
        out = self.fc(out)
        return out

class NeuralLatentLightEstimator(nn.Module):
    """ Estimate Neural Latent lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(NeuralLatentLightEstimator, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=None)
        self.conv1    = get_conv(384, 128, kernel_size=1, stride=1)
        self.conv2    = get_conv(128, 64, kernel_size=1, stride=1)
        self.conv3    = get_conv(64, 64, kernel_size=3, padding=1)
        self.conv4    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out

class ShadingCorrectNess(nn.Module):
    """ Correct generated shading with neural light
    """
    def __init__(self):
        super(ShadingCorrectNess, self).__init__()
        self.conv1 = get_conv(6, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        return out

class ReconstructImage(nn.Module):
    """ Reconstruct Image from shading and albedo
    """
    def __init__(self):
        super(ReconstructImage, self).__init__()
    
    def forward(self, shading, albedo):
        return shading * albedo
        
class SfsNetPipeline(nn.Module):
    """ SfSNet Pipeline
    """
    def __init__(self, conv_model, normal_residual_model, albedo_residual_model,
                    light_estimator_model, normal_gen_model, albedo_gen_model, shading_model,
                    neural_light_model, shading_correctness_model, image_recon_model):
        super(SfsNetPipeline, self).__init__()
        self.conv_model = conv_model
        self.normal_residual_model = normal_residual_model
        self.albedo_residual_model = albedo_residual_model
        self.light_estimator_model = light_estimator_model
        self.normal_gen_model      = normal_gen_model
        self.albedo_gen_model      = albedo_gen_model
        self.shading_model         = shading_model
        self.neural_light_model    = neural_light_model
        self.shading_correctness_model = shading_correctness_model
        self.image_recon_model     = image_recon_model

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
        # 3 d. Estimate Neural Light for residual correctness
        predicted_neural_light = self.neural_light_model(all_features)

        # 4. Generate shading
        out_shading = self.shading_model(denorm(predicted_normal), predicted_sh)

        # 5. Correct shading with Neural Light
        shading_light = torch.cat((out_shading, predicted_neural_light), dim=1)
        corrected_shading = self.shading_correctness_model(shading_light)

        # 6. Reconstruction of image
        out_recon = self.image_recon_model(corrected_shading, denorm(predicted_albedo))
                    
        return predicted_normal, predicted_albedo, predicted_sh, out_shading, corrected_shading, out_recon
