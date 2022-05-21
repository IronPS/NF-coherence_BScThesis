import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

from types import SimpleNamespace
import random
import gc

import numpy as np
import torch

import matplotlib.pyplot as plt

from tcc.eval import ModelLoader
from tcc.util import directory as dir
from tcc.util import loader_functions as lf

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
    parser = argparse.ArgumentParser(description='Plot a trained model losses')
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

    checkpoint_frequency = 8192 / params.batch_size

    fig, ax = plt.subplots()
    last_train_x = 0
    last_test_x = 0
    for ((epoch_train, train_losses), (epoch_test, test_losses)) in zip(gmLoader['TrainLosses'], gmLoader['TestLosses']):
        assert epoch_train == epoch_test

        next_train_x = last_train_x + train_losses.shape[0]
        train_range = np.arange(last_train_x, next_train_x)
        ax.plot(train_range, train_losses.cpu().numpy(), color='blue', label='Train loss')
        
        # TODO there is a mismatch between number of train loss batch number and test batch number for some reason unknown
        # The overall idea of the plot still remains
        next_test_x = last_test_x + test_losses.shape[0] * checkpoint_frequency
        test_range = np.arange(last_test_x, next_test_x, checkpoint_frequency)
        if test_range[-1] > train_range[-1]:
            test_range = test_range[:-1]
            test_losses = test_losses[:-1]
            next_test_x = last_test_x + test_losses.shape[0] * checkpoint_frequency

        ax.scatter(test_range, test_losses.cpu().numpy(), color='red', s=8, label='Test loss', zorder=10)
        
        ax.axvline(last_train_x, color='black', linestyle='dashed', alpha=0.3, label='Epoch start')
        
        last_train_x = next_train_x
        last_test_x = next_test_x
        # if epoch_train == 1:
        #     break

    ax.set_xlabel("# Batches")
    ax.set_ylabel("NLL Loss")

    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    fig.savefig(eval_params.output_folder + '/losses.png')
