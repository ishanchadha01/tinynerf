from utils import *

from os import truncate
import numpy as np
import os, sys
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


def posenc(x, L_embed):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.concat(rets, -1)


class NerfModel(nn.Module):
    def __init__(self, input_shape, D=8, W=256):
        super().__init__()
        self.feats_ = [
            nn.Linear(in_features=input_shape, out_features=W),
            nn.ReLU()
        ]
        for i in range(D-1):
            self.feats_ += [
                nn.Linear(in_features=W, out_features=W),
                nn.ReLU()
            ]
        # self.feats_.append(nn.Linear(W, 4))
        self.final = nn.Linear(W, 4)
        self.model = nn.Sequential(*self.feats_) # issue, i dont think feats are called

    def forward(self, x):
        inp = x
        x = self.model(x)
        x = self.final(x)
        return x


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3,:3], -1)
    rays_o = torch.Tensor(c2w[:3, -1]).unsqueeze(0).unsqueeze(0).expand_as(rays_d) #take last col of c2w (cam origin) and repeat it for each ray
    return rays_o, rays_d


def render_rays(model, rays_o, rays_d, near, far, N_samples, L_embed, rand=False):
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples)
    # if rand:
    #     z_vals += torch.rand(list(rays_o.shape[:-1]) + [N_samples]) * (far - near) / N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = pts.view(-1, 3)
    pts_flat = posenc(pts_flat, L_embed)
    chunk = 1024*32
    raw = torch.cat([model(pts_flat[i:i+chunk]) for i in range(0, pts_flat.shape[0], chunk)], 0)
    raw = raw.view(list(pts.shape[:-1]) + [4]) #TODO: issue here!

    # Compute opacities and colors
    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full_like(z_vals[..., :1], 1e10)], -1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, -1)

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map






def main():

    ### Get LLFF Data (COLMAP)

    factor=8 # downsample factor for llff imgs
    datadir = './gtri_jetski'
    llffhold = 8 # 1/N images taken as testset
    images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=True)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('Loaded llff', images.shape,
        render_poses.shape, hwf, datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = torch.Tensor(np.arange(images.shape[0])[::llffhold])

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    near = 0.
    far = 1.
    print('NEAR FAR', near, far)

    ### Visualizing data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    np.random.seed(0)
    DEBUG = False
    L_embed = 6

    testimg, testpose = torch.Tensor(images[11]), torch.Tensor(poses[11])
    images = images[:100,...,:3]
    poses = poses[:100]

    plt.imshow(testimg)
    plt.show()


    ### Run model
    print("Creating model")
    model = NerfModel(3 + 3*2*L_embed)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    N_samples = 64
    N_iters = 1000
    psnrs = []
    iternums = []
    i_plot = 25
    H = 160
    W = 90
    focal=1536

    t = time.time()
    model.train()
    for i in tqdm(range(N_iters + 1)):
        img_i = np.random.randint(images.shape[0])
        target = torch.tensor(images[img_i], dtype=torch.float32)
        pose = torch.tensor(poses[img_i], dtype=torch.float32)
        rays_o, rays_d = get_rays(H, W, focal, pose)

        optimizer.zero_grad()
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, L_embed=L_embed, rand=True)
        loss = torch.mean((rgb - target) ** 2)
        loss.backward()
        optimizer.step()

        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()

        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, L_embed=L_embed)
        loss = torch.mean(torch.square(rgb - testimg))
        psnr = -10. * torch.log10(loss)

        psnrs.append(psnr.item())
        print(f"PSNR for iter {i}: {psnrs[-1]}")
        iternums.append(i)
    


if __name__=='__main__':
    main()