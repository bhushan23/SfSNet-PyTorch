import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

    if classname.find('Linear') != -1:
    	init.normal(m.weight)
    	init.constant(m.bias,1)



class conv3x3(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv1x1(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv7x7(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(conv7x7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, stride=1, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlk(nn.Module):

    def __init__(self, ch):
        super(ResBlk, self).__init__()
        #self.bn1 = nn.BatchNorm2d(ch),
        #self.relu1 = nn.ReLU(inplace=True),
        #self.conv1 = nn.Conv2d(ch, ch, 3, padding=1),
        #self.bn2 = nn.BatchNorm2d(ch),
        #self.relu2 = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(ch, ch, 3, padding=1),

        self.res = nn.Sequential(
        	nn.BatchNorm2d(ch),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(ch, ch, 3, stride=1, padding=1),
        	nn.BatchNorm2d(ch),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(ch, ch, 3, stride=1, padding=1)
        	)


    def forward(self,x):
        residual = x
        #out = F.relu(self.bn1(x))
        #out = self.conv1(out)
        #out = F.relu(self.bn2(out))
        #out = self.conv2(out)
        out = self.res(x)

        out += residual

        return out
        

class SfSNet(nn.Module):
    def __init__(self):
    	super(SfSNet, self).__init__()
    	self.conv1=conv7x7(3,64)
    	self.conv2=conv3x3(64,128)
    	self.conv3=nn.Conv2d(128,128, 3, stride=2,padding=1)

    	#Normal
    	self.nres1=ResBlk(128)
    	self.nres2=ResBlk(128)
    	self.nres3=ResBlk(128)
    	self.nres4=ResBlk(128)
    	self.nres5=ResBlk(128)
    	self.nreso=nn.Sequential(
	            nn.BatchNorm2d(128),
	            nn.ReLU(inplace=True),
	        )
    	self.nup=nn.Upsample(scale_factor=2,mode='bilinear')
    	self.nconv1=conv1x1(128,128)
    	self.nconv2=conv3x3(128,64)
    	self.nout=nn.Conv2d(64,3, 1, stride=1,padding=0)

    	#Albedo
    	self.ares1=ResBlk(128)
    	self.ares2=ResBlk(128)
    	self.ares3=ResBlk(128)
    	self.ares4=ResBlk(128)
    	self.ares5=ResBlk(128)
    	self.areso=nn.Sequential(
	            nn.BatchNorm2d(128),
	            nn.ReLU(inplace=True),
	        )
    	self.aup=nn.Upsample(scale_factor=2,mode='bilinear')
    	self.aconv1=conv1x1(128,128)
    	self.aconv2=conv3x3(128,64)
    	self.aout=nn.Conv2d(64,3, 1, stride=1,padding=0)

    	#Light
    	self.lconv=conv1x1(384,128)
    	self.l0=nn.AvgPool2d(64,stride=1,padding=0)
    	self.lout=nn.Linear(128,27)


    def forward(self,x):
	
        c1=self.conv1(x)
        c2=self.conv2(c1)
        c3=self.conv3(c2)

        # print(c3.size())

        #Normal
        nr1=self.nres1(c3)
        nr2=self.nres2(nr1)
        nr3=self.nres3(nr2)
        nr4=self.nres4(nr3)
        nr5=self.nres5(nr4)
        nro=self.nreso(nr5)

        nup=self.nup(nro)
        nc1=self.nconv1(nup)
        nc2=self.nconv2(nc1)
        nout=self.nout(nc2)


        #Albedo
        ar1=self.ares1(c3)
        ar2=self.ares2(ar1)
        ar3=self.ares3(ar2)
        ar4=self.ares4(ar3)
        ar5=self.ares5(ar4)
        aro=self.areso(ar5)

        aup=self.aup(aro)
        ac1=self.aconv1(aup)
        ac2=self.aconv2(ac1)
        aout=self.aout(ac2)

        #Light
        l1=torch.cat([nro,aro],dim=1)
        l2=torch.cat([c3,l1],dim=1)
        lc1=self.lconv(l2)
        l0=self.l0(lc1)
        lout=self.lout(torch.squeeze(l0))

        return nout, aout, lout

#lout is float tensor of size 27. just a simple column
	

class SfSNet_Base_Pipeline(nn.Module):
  def __init__(self, shading_model, image_recon_model):
    super(SfSNet_Base_Pipeline, self).__init__()
    self.shading_model = shading_model
    self.image_recon_model = image_recon_model
    self.model = SfSNet()
    
  def get_face(self, sh, normal, albedo):
    shading = self.shading_model(normal, sh)
    recon   = self.image_recon_model(shading, albedo)
    return recon

  def forward(self, x):
    normal, albedo, sh = self.model(x)

    out_shading = self.shading_model(normal, sh)
    out_recon = self.image_recon_model(out_shading, albedo)

    return normal, albedo, sh, out_shading, out_recon
