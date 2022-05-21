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
from tqdm import tqdm

from tcc.eval import ModelLoader
from tcc.util import directory as dir
from tcc.util import loader_functions as lf
from tcc.data import CelebALoader

import argparse

def print_models(models_dir, show=False):
    from datetime import datetime

    models_paths = [m for m in dir.list_dir(models_dir, dir.ListDirPolicyEnum.DIRS_ONLY) if 'Glow' in m]

    if show:
        for i, p in enumerate(models_paths):
            print("- {}: '{}' -- {}".format(i, p.split('/')[-1], datetime.fromtimestamp(os.path.getmtime(p))))
        
        exit(0)

    return models_paths


def parse():
    parser = argparse.ArgumentParser(description='Generate latent space vectors of the test and evaluation partititions of the CelebA dataset using a trained NF-based model')
    parser.add_argument('-d' ,'--dpath',
        required=True,
        type=str,
        help='Path to the models\' saved data'
    )
    parser.add_argument('-l' ,'--list',
        required=False,
        action='store_true',
        help='Lists all models\' saved data'
    )
    parser.add_argument('-m' ,'--model',
        required=False,
        type=int,
        help='The integer identifier of a model in the path'
    )
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Directory where the generated images will be saved'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )

    params = parser.parse_args()
    
    params.dpath = os.path.expandvars(params.dpath)
    assert os.path.exists(params.dpath), "Data path '{}' does not exist".format(params.dpath)
    if not os.path.isabs(params.dpath):
        params.dpath = os.path.abspath('.' + os.path.sep + params.dpath)

    models = print_models(params.dpath, params.list)

    if not os.path.isabs(params.output_folder):
        params.output_folder = os.path.abspath('.' + os.path.sep + params.output_folder)
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    os.makedirs(params.output_folder + '/test', exist_ok=True)
    os.makedirs(params.output_folder + '/eval', exist_ok=True)

    
    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    params.model_path = models[params.model]

    return params

if __name__ == "__main__":
    eval_params = parse()
    
    def load_glow(path, device=None):
        return torch.load(path, map_location=device)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaders_dict = {
        'Params': lf.json_load_fn,
        'TrainLosses': lf.torch_tensor_load_fn,
        'TestLosses': lf.torch_tensor_load_fn,
        'TrainedFlow': lambda x: load_glow(x, device=device),
    }
    gmLoader = ModelLoader(eval_params.model_path, loaders_dict=loaders_dict)
    print(gmLoader)

    params = gmLoader['Params'][0]
    for _, p in params.iterrows():
        print(p)

    params = SimpleNamespace(**params.iloc[0].to_dict())
    params.dataset_path = "{}{}..{}..{}data{}".format(FPATH, *[os.sep]*4) + params.dataset_path.split('data'+os.sep)[-1] 


    loader_params = {
        'batch_size': params.batch_size,
        'shuffle': False,
        # 'collate_fn': lambda x: default_collate(x).to(device),
        'num_workers': 0,
        'pin_memory': device.type.upper() == "CUDA"
    }

    test_gen = CelebALoader(params, "test", **loader_params)

    model = gmLoader['TrainedFlow'][len(gmLoader['TrainedFlow'])-1]

    # print(model._distribution._log_prob((model.mean_() - 2*model.std_()).float(), None).item())

    with torch.no_grad():
        for x, idx in tqdm(test_gen):
            x = x.to(device).float()
            lats = model.transform_to_noise(x)
            for l, i in zip(lats, idx):
                torch.save(l, eval_params.output_folder + '/test/' + str(i.item()) + '.pt')

    validation_gen = CelebALoader(params, "eval", **loader_params)

    with torch.no_grad():
        for x, idx in tqdm(validation_gen):
            x = x.to(device).float()
            lats = model.transform_to_noise(x)
            for l, i in zip(lats, idx):
                torch.save(l, eval_params.output_folder + '/eval/' + str(i.item()) + '.pt')
