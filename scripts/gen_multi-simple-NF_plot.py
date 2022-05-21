

# Based in
# https://github.com/bayesiains/nflows/blob/master/examples/moons.ipynb

import os

import numpy as np
import random

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from tqdm import tqdm

import matplotlib.pyplot as plt


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Generate multiple plots of a simple NF-based model')
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Folder to which output all images'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )
    params = parser.parse_args()

    if not os.path.isabs(params.output_folder):
        params.output_folder = os.path.abspath('.' + os.path.sep + params.output_folder)
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    return params

if __name__ == "__main__":
    params = parse()

    x = np.arange(-np.pi, np.pi, step=1./360)
    x = np.stack((x, np.sin(x))).T
    n = x.shape[0]
    plt.scatter(x[:, 0], x[:, 1], alpha=0.01)
    plt.savefig(params.output_folder + '/001-initial-data')
    plt.clf()

    means = np.sort(np.random.uniform(low=-np.pi, high=np.pi, size=n))
    variance = 0.1 * np.random.standard_normal(size=n)
      
    X = np.stack((means, np.sin(means) + variance)).T
    plt.scatter(X[:, 0], X[:, 1], alpha=0.05)
    plt.savefig(params.output_folder + '/002-noisy-data')
    plt.clf()

    def get_num_images(num_iter, plot_freq=500):
        has_rest = num_iter % plot_freq == 0
        n_images = num_iter//plot_freq + (1 if has_rest else 0)
        
        return n_images

    num_iter = 10000
    plot_freq = 500
    num_layers = 10

    flows = []

    print("Training flows...")
    n_images = get_num_images(num_iter, plot_freq=plot_freq)
    fig, axes = plt.subplots(num_layers, n_images, figsize=(num_layers*4, n_images*4))
    for nl in range(1, num_layers+1):
        print("{}/{}".format(nl, num_layers))
        ax_idx = (nl-1, 0)
        
        transforms = []
        for _ in range(nl):
            transforms += [ReversePermutation(features=2), MaskedAffineAutoregressiveTransform(features=2, hidden_features=4)]

        transform = CompositeTransform(transforms)

        base_dist = StandardNormal(shape=[2])
        flow = Flow(transform, base_dist)
        optimizer = optim.Adam(flow.parameters())

        for i in tqdm(range(num_iter)):
            x = torch.tensor(X, dtype=torch.float32)
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x).mean()
            loss.backward()
            optimizer.step()

            if (i + 1) % plot_freq == 0:
                xline = torch.linspace(-np.pi, np.pi, steps=100)
                yline = torch.linspace(-1.25, 1.25, steps=100)
                xgrid, ygrid = torch.meshgrid(xline, yline, indexing='xy')
                xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

                with torch.no_grad():
                    zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

                axes[ax_idx].contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
                axes[ax_idx].set_title("n_layers: \'{}\', iter: \'{}\'".format(nl, i + 1))

                ax_idx = ax_idx[0], ax_idx[1]+1

        if ax_idx[1] != n_images:
            xline = torch.linspace(-np.pi, np.pi, steps=100)
            yline = torch.linspace(-1.25, 1.25, steps=100)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

            with torch.no_grad():
                zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

            axes[ax_idx].contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            axes[ax_idx].set_title("n_layers: \'{}\', iter: \'{}\'".format(nl, i + 1))

            ax_idx = ax_idx[0], ax_idx[1]+1
                
        flows.append(flow)

    plt.tight_layout()
    plt.savefig(params.output_folder + '/003-flow-densities')
    plt.clf()
    plt.close(fig)

    print("Generating samples...")
    fig, axes = plt.subplots(1, len(flows), figsize=(len(flows)*2, 2))

    n_samples = 250
    for i, flow in tqdm(enumerate(flows)):
        x = flow.sample(n_samples).detach().numpy()

        axes[i].scatter(x[:, 0], x[:, 1], alpha=0.1)
        axes[i].set_title("n_layers: {}".format(i+1))
        
    plt.tight_layout()
    plt.savefig(params.output_folder + '/004-{}samples'.format(n_samples))
    plt.clf()
    plt.close(fig)


    print("Creating transformation sequence...")
    cmap = plt.get_cmap('gnuplot')

    max_transforms = max([len(flow._transform._transforms) for flow in flows])
    rows, cols = len(flows), max_transforms+1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

    for i, flow in enumerate(flows):
        axes[i,0].scatter(X[:, 0], X[:, 1], alpha=0.05, color=cmap(np.arange(0,1,(1./X.shape[0]))))
        axes[i,0].set_title("Orig. data. # Layers: \'{}\'".format(i+1))
        transforms_list = flow._transform._transforms

        t_x = X
        for j, t in enumerate(transforms_list):
            j += 1
            
            t_x = torch.tensor(t_x, dtype=torch.float32)
            t_x1, ldj = t(t_x)
                       
            t_x = t_x1
            t_x = t_x.detach().numpy()
            axes[i,j].scatter(t_x[:, 0], t_x[:, 1], alpha=0.05, color=cmap(np.arange(0,1,(1./X.shape[0]))))
            axes[i,j].set_title("# Transform: \'{}\'".format(j))
            
    plt.tight_layout()
    plt.savefig(params.output_folder + '/005-transform-sequence')
    plt.clf()
    plt.close(fig)


    
