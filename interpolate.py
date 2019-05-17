from models import *
from utils import save_image

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import os
import argparse


IMAGE_SIZE = 128

def interpolate(model_dir, input_path, output_path):
  use_cuda = torch.cuda.is_available()

  os.system('mkdir -p {}'.format(output_path))

  # Load images
  transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
            ])

  img_dataset = torchvision.datasets.ImageFolder(input_path, transform=transform)
  dl          = DataLoader(img_dataset, batch_size=1)

  print('Data size:', len(dl))

  # Load model
 
  sfs_net_model         = SfsNetPipeline()
  if use_cuda:
      sfs_net_model = sfs_net_model.cuda()
  
  sfs_net_model.load_state_dict(torch.load(model_dir + 'sfs_net_model.pkl'))

  for bix, (data, _) in enumerate(dl):
    if use_cuda:
      data = data.cuda()

    normal, albedo, sh, shading, recon = sfs_net_model(data)
    output_dir = output_path + str(bix)

    # normal = normal * 128 + 128
    # normal = normal.clamp(0, 255) / 255
    save_image(data, path=output_dir+'_face.png')
    save_image(normal, path=output_dir+'_normal.png')
    save_image(albedo, path=output_dir+'_albedo.png')
    save_image(shading, path=output_dir+'_shading.png')
    save_image(recon, path=output_dir+'_recon.png')
    sh = sh.cpu().detach().numpy()
    np.savetxt(output_dir+'_light.txt', sh, delimiter='\t')


def main():
    parser = argparse.ArgumentParser(description='SfSNet - Interpolation')

    parser.add_argument('--data', type=str, default='../data/interpolation-input/faces/',
                        help='interpolation input')
    parser.add_argument('--load_model', type=str, default=None,
                        help='load model from')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Interpolation output path')

    args = parser.parse_args()
    model_dir  = args.load_model
    data_dir   = args.data
    output_dir = args.output_dir

    # load Synthetic trained model only
    # model_path = model_dir + 'Synthetic_Train/checkpoints/'
    # output_dir_syn = output_dir + '/Synthetic_Train_Interpolation/'
    # interpolate(model_path, data_dir, output_dir_syn)

    # load Mix Data trained model
    model_path = model_dir + 'Mix_Training/checkpoints/'
    output_dir_mix = output_dir + '/Mix_Train_Interpolation/'
    interpolate(model_path, data_dir, output_dir_mix)

if __name__ == '__main__':
    main()
