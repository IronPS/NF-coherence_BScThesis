import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

from types import SimpleNamespace
import gc
from joblib import dump, load

import numpy as np
import torch
import random
import pandas as pd
from tqdm import tqdm
from natsort import natsort_keygen, natsorted

from tcc.distribution import NormalGamma, DiagonalGaussian

from tcc.util import directory as dir
from tcc.util import loader_functions as lf

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Fits normal-gamma distributions for each attribute class.')
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
        help='Directory where the generated distributions will be saved'
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
    assert os.path.exists(params.celeba_metadata_path), "Data path '{}' does not exist".format(params.celeba_metadata_path)
    if not os.path.isabs(params.celeba_metadata_path):
        params.celeba_metadata_path = os.path.abspath('.' + os.path.sep + params.celeba_metadata_path)

    if not os.path.isabs(params.output_folder):
        params.output_folder = os.path.abspath('.' + os.path.sep + params.output_folder)
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    os.makedirs(params.output_folder + '/distribution/', exist_ok=True)
    os.makedirs(params.output_folder + '/lprobs/', exist_ok=True)

    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    return params

def get_data_info(params, partition=1, max_samples=None):
    """
        partition: 1 for test, 2 for validation
    """

    def replace_fn(x):
        return x.replace(".jpg", '')
    transform = np.vectorize(replace_fn)
        
    ids = pd.read_csv(params.celeba_metadata_path + "/list_eval_partition.txt", names=["image", "partition"], header=None, delimiter=' ') \
        .groupby(by="partition")["image"] \
        .get_group(partition) # test partition
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

    return data[:max_samples]

def get_latents_tensor(params, partition=1, max_samples=None):
    folder_complement = '/test/'
    if partition == 1:
        folder_complement = '/test/'
    elif partition == 2:
        folder_complement = '/eval/'

    lat_files = natsorted(['.'.join(m.split('/')[-1].split('.')[:-1]) for m in dir.list_dir(params.latents_path + folder_complement, dir.ListDirPolicyEnum.FILES_ONLY)])[:max_samples]
    
    tensors = []
    print("Collecting tensors...")
    for l in tqdm(lat_files):
        tensors.append(torch.load(params.latents_path + folder_complement + l + '.pt', map_location='cpu').unsqueeze(0))

    print("Concatenating tensors...")
    tensors = torch.cat(tensors)
    
    return tensors, lat_files

if __name__ == "__main__":
    eval_params = parse()

    max_samples = None
    test_data_info = get_data_info(eval_params, partition=1, max_samples=max_samples)
    eval_data_info = get_data_info(eval_params, partition=2, max_samples=max_samples)
    print("Data columns:", flush=True)
    print(str(test_data_info.columns))

    test_lats, test_names = get_latents_tensor(eval_params, partition=1, max_samples=max_samples)
    eval_lats, eval_names = get_latents_tensor(eval_params, partition=2, max_samples=max_samples)

    cols = test_data_info.columns[(test_data_info.columns != 'image')]

    # cols = ['Male', 'Smiling', 'Young']
    print("Initiating distribution fitting...", flush=True)
    items = []
    for c in tqdm(cols):
        # try:
            probs_df = pd.DataFrame()

            mask = test_data_info[c] == 1
            eval_mask = eval_data_info[c] == 1
            
            X = test_lats[mask]
            ng_model = NormalGamma(m=0, l=1, a=0.001, b=0.001).fit(X)

            # pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision() * (ng_model._l + 1)))
            pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision()))
            
            probs = pred_dg_model.lpdf(eval_lats)

            probs_df['positiveAttr'] = eval_mask
            
            probs_df['p_lprobs'] = probs.cpu().numpy()

            dump(ng_model, eval_params.output_folder + '/distribution/p-{}_posterior.joblib'.format(c))


            n_obs_used_before = np.count_nonzero(mask)
            mask = ~mask

            X = test_lats[mask]
            X = X[:n_obs_used_before]
            ng_model = NormalGamma(m=0, l=1, a=0.001, b=0.001).fit(X)

            # pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision() * (ng_model._l + 1)))
            pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision()))
            
            probs = pred_dg_model.lpdf(eval_lats)
            
            probs_df['n_lprobs'] = probs.cpu().numpy()

            dump(ng_model, eval_params.output_folder + '/distribution/n-{}_posterior.joblib'.format(c))
        
            probs_df.to_csv(eval_params.output_folder + '/lprobs/{}-lprobs.csv'.format(c))
        # except Exception as e:
        #     print("Error:", sys.exc_info()[0])
        #     print("Continuing...", flush=True)

    print("Distributions learned! Saving metadata...", flush=True)
    # logprobs_df = pd.DataFrame(items)

    # logprobs_df.to_csv(eval_params.output_folder + '/gaussian_dist_logprobs.csv')
    test_data_info.to_csv(eval_params.output_folder + '/test_data_info_gaussian_dist.csv')
    eval_data_info.to_csv(eval_params.output_folder + '/eval_data_info_gaussian_dist.csv')

    print("Done.", flush=True)