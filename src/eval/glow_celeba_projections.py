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
from natsort import natsort_keygen, natsorted

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tcc.util import directory as dir
from tcc.util import loader_functions as lf

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train dimensionality reductions models from a set of latent space vectors and generates their 2D embeddings')
    parser.add_argument('--celeba_metadata_path',
        required=True,
        type=str,
        help='Path to the CelebA metadata directory'
    )
    parser.add_argument('--latents_path',
        required=True,
        type=str,
        help='Path to the data latents'
    )
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Directory where the generated images will be saved'
    )
    parser.add_argument('--pca_path',
        required=False,
        type=str,
        default='',
        help='Path for the data PCA embedding, if available'
    )
    parser.add_argument('--tsne',
        required=False,
        action='store_true',
        help='Generates 2D projections using t-SNE'
    )
    parser.add_argument('--umap',
        required=False,
        action='store_true',
        help='Generates 2D projections using UMAP'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )

    params = parser.parse_args()
    
    params.latents_path = os.path.expandvars(params.latents_path)
    assert os.path.exists(params.latents_path), "Data path '{}' does not exist".format(params.latents_path)
    if not os.path.isabs(params.latents_path):
        params.latents_path = os.path.abspath('.' + os.path.sep + params.latents_path)
    
    params.celeba_metadata_path = os.path.expandvars(params.celeba_metadata_path)
    assert os.path.exists(params.celeba_metadata_path), "Data path '{}' does not exist".format(params.latents_path)
    if not os.path.isabs(params.celeba_metadata_path):
        params.celeba_metadata_path = os.path.abspath('.' + os.path.sep + params.celeba_metadata_path)

    if not os.path.isabs(params.output_folder):
        params.output_folder = os.path.abspath('.' + os.path.sep + params.output_folder)
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)
    
    if params.pca_path:
        if not os.path.exists(params.pca_path):
            print("Could not find the file '{}'".format(params.pca_path))
            exit(1)

    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    return params

def get_data_info(params):
    def replace_fn(x):
        return x.replace(".jpg", '')
    transform = np.vectorize(replace_fn)
        
    ids = pd.read_csv(params.celeba_metadata_path + "/list_eval_partition.txt", names=["image", "partition"], header=None, delimiter=' ') \
        .groupby(by="partition")["image"] \
        .get_group(1) # test partition
    ids = transform(ids)
    ids = pd.DataFrame({'image': ids}).set_index('image')

    df_attrs = pd.read_csv(params.celeba_metadata_path + '/list_attr_celeba.txt', header=1, delim_whitespace=True)
    df_attrs = df_attrs.rename_axis('image')
    df_attrs = df_attrs.reset_index()
    df_attrs.image = transform(df_attrs.image)
    df_attrs = df_attrs.set_index('image')
    data = df_attrs.join(ids, how='inner', on='image').reset_index()

    data.sort_values(
        by='image', 
        key=natsort_keygen(),
        inplace=True
    )

    return data

def get_latents_tensor(params, ignore_latents=False):
    lat_files = natsorted(['.'.join(m.split('/')[-1].split('.')[:-1]) for m in dir.list_dir(params.latents_path, dir.ListDirPolicyEnum.FILES_ONLY)])
    
    tensors = []
    if not ignore_latents:
        print("Collecting tensors...")
        for l in tqdm(lat_files):
            tensors.append(torch.load(params.latents_path + '/' + l + '.pt', map_location='cpu').unsqueeze(0))
        
        print("Concatenating tensors...")
        tensors = torch.cat(tensors)
    
    return tensors, lat_files

if __name__ == "__main__":
    eval_params = parse()

    data = get_data_info(eval_params)
    print("Data columns:")
    print(str(data.columns))


    lats, names = get_latents_tensor(eval_params, ignore_latents=eval_params.pca_path)
    if not eval_params.pca_path:
        print("Latents shape:", lats.shape)

        print("Applying PCA...")
        pca_model = PCA(n_components=0.8, svd_solver='full').fit(lats)
        print("Found {} components".format(pca_model.n_components_))
        print("PCA explained variances")
        print(str(pca_model.explained_variance_ratio_))
        print("Total variance explained:", np.sum(pca_model.explained_variance_ratio_))
    
        print()
        print("Transforming data with PCA model...")
        pca_embedding = pca_model.transform(lats)
    
    else:
        print("Loading PCA embedding...")
        pca_embedding = torch.load(eval_params.pca_path, map_location='cpu')
        print("Embedding shape:", pca_embedding.shape)

    print()
    if eval_params.tsne:
        print("Applying t-SNE")
        tsne_embedding = TSNE(n_components=2, perplexity=1500, learning_rate=150, n_iter=2500, init='random', verbose=1).fit_transform(pca_embedding)


    if eval_params.umap:
        print("Applying UMAP...")
        umap_embedding = UMAP(n_components=2, n_neighbors=4, min_dist=0.25, metric='euclidean').fit_transform(pca_embedding)

    torch.save(pca_embedding, eval_params.output_folder + '/pca_embedding.pt')
    
    if eval_params.tsne:
        torch.save(tsne_embedding, eval_params.output_folder + '/t-sne_embedding.pt')
    
    if eval_params.umap:
        torch.save(umap_embedding, eval_params.output_folder + '/umap_embedding.pt')

    pd.DataFrame(names, columns=['name']).to_csv(eval_params.output_folder + '/embedding_names.csv')
    data.to_csv(eval_params.output_folder + '/data_info.csv')
