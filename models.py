import torch
import torch.nn as nn

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

        print(c, len(sh))
        assert(c == len(sh))

        print(Y1.shape, Y2.shape, Y3.shape, Y4.shape, Y5.shape, Y6.shape, Y7.shape, Y8.shape, Y9.shape)
        shading = torch.zeros(b, c, h, w)
        for j in range(c):
            l = sh[j]
            print(l.shape, l[:, 0])
            shading[:, j, :, :] += Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                                  Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                                  Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

        return shading

