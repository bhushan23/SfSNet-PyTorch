import matplotlib.pyplot as plt
import torchvision

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_image(pic, path=None):
    pic = denorm(pic)
    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)