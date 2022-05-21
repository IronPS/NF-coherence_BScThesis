import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

from types import SimpleNamespace
import gc

import numpy as np
import torch
import random
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from tcc.util import directory as dir
from tcc.util import loader_functions as lf

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Plots previously calculated projections')
    parser.add_argument('--results_path',
        required=True,
        type=str,
        help='Path to the directory containing the results of the projections'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )
    parser.add_argument('--tsne',
        required=False,
        action='store_true',
        help='Generates plots for t-SNE'
    )
    parser.add_argument('--umap',
        required=False,
        action='store_true',
        help='Generates plots for UMAP'
    )

    params = parser.parse_args()
    
    params.results_path = os.path.expandvars(params.results_path)
    assert os.path.exists(params.results_path), "Data path '{}' does not exist".format(params.results_path)
    if not os.path.isabs(params.results_path):
        params.results_path = os.path.abspath('.' + os.path.sep + params.results_path)
    
    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    return params

def get_tsne_embedding(params):
    embedding = torch.load(params.results_path + '/t-sne_embedding.pt', map_location='cpu')
    return embedding

def get_umap_embedding(params):
    embedding = torch.load(params.results_path + '/umap_embedding.pt', map_location='cpu')
    return embedding

def get_data_info(params):
    embedding_names = pd.read_csv(params.results_path + '/embedding_names.csv')
    data_info = pd.read_csv(params.results_path + '/data_info.csv')
    return embedding_names, data_info

if __name__ == "__main__":
    eval_params = parse()

    print("Loading data info...")
    embedding_names, data_info = get_data_info(eval_params)

    tsne_embedding = None
    umap_embedding = None

    if eval_params.tsne:
        print("Loading t-SNE embedding...")
        tsne_embedding = get_tsne_embedding(eval_params)

    if eval_params.umap:
        print("Loading UMAP embedding...")
        umap_embedding = get_umap_embedding(eval_params)

    print("Plotting...")

    embeddings = [tsne_embedding,  umap_embedding]
    embedding_fnames = ["tsne", "umap"]

    for embedding_fname, embedding in zip(embedding_fnames, embeddings):
        if type(embedding) == type(None):
            continue

        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

        x_min, x_max = x_min - x_min*0.01, x_max + x_max*0.01
        y_min, y_max = y_min - y_min*0.01, y_max + y_max*0.01

        cols = data_info.columns[(data_info.columns != 'image')]
        for c in cols:
            mask = data_info[c] == 1
            mask_color = 'blue'
            nmask_color = 'black'
            if mask.shape[0]/2 > mask.count():
                mask = ~mask
                mask_color = nmask_color
                nmask_color = 'blue'
            nmask = ~mask

            fig, ax = plt.subplots()
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=mask_color, alpha=0.75)
            ax.scatter(embedding[nmask, 0], embedding[nmask, 1], c=nmask_color, alpha=0.2)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            fig.savefig(eval_params.results_path + '/{}_{}.png'.format(embedding_fname, c), dpi=fig.dpi)
            
            plt.close(fig)
            del fig
