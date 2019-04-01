import matplotlib.pyplot as plt
import torchvision
from PIL import Image

def applyMask(input_img, mask):
    return input_img * mask

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_image_grid(pic, denormalize=True, mask=None):
    if denormalize:
        pic = denorm(pic)
    
    if mask is not None:
        pic = pic * mask

    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def save_image(pic, denormalize=True, path=None, mask=None):
    ndarr = get_image_grid(pic, denormalize=denormalize, mask=mask)    
    
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)

def wandb_log_images(wandb, img, mask, caption, step, log_name, path=None, denormalize=True):
    ndarr = get_image_grid(img, denormalize=denormalize, mask=mask)

    # save image if path is provided
    if path is not None:
        im = Image.fromarray(ndarr)
        im.sve(path)

    wimg = wandb.Image(ndarr, caption=caption)
    wandb.log({log_name: wimg})