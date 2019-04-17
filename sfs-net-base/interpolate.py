from models import *
from utils import save_image

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import numpy as np

def interpolate(model_dir, input_path, output_path):
  use_cuda = torch.cuda.is_available()

  # Load images
  img_dataset = torchvision(.datasets.ImageFolder(input_path)
  dl          = DataLoader(img_dataset, batch_size=1)

  # Load model
  conv_model            = baseFeaturesExtractions()
  normal_residual_model = NormalResidualBlock()
  albedo_residual_model = AlbedoResidualBlock()
  light_estimator_model = LightEstimator()
  normal_gen_model      = NormalGenerationNet()
  albedo_gen_model      = AlbedoGenerationNet()
  shading_model         = sfsNetShading()
  image_recon_model     = ReconstructImage()
  
  sfs_net_model         = SfsNetPipeline(conv_model, normal_residual_model, albedo_residual_model, \
                                          light_estimator_model, normal_gen_model, albedo_gen_model, \
                                          shading_model, image_recon_model)
  if use_cuda:
      sfs_net_model = sfs_net_model.cuda()
  
  sfs_net_model.load_state_dict(torch.load(model_dir + 'sfs_net_model.pkl'))

  for bix, data in enumerate(dl):
    if use_cuda:
      data = data.cuda()

    normal, albedo, sh, shading, recon = sfs_net_model(data)

    save_image(data, path=str(bix)+'_face.png')
    save_image(normal, path=str(bix)+'_normal.png')
    save_image(albedo, path=str(bix)+'_albedo.png')
    save_image(shading, path=str(bix)+'_shading.png')
    save_image(recon, path=str(bix)+'_recon.png')
    sh = sh.cpu().detach().numpy()
    np.savetxt(str(bix)+'_light.txt', delimiter='\t')   