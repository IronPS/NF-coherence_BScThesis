
import os
import sys

from natsort import natsorted

FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../src/'))

from tqdm import tqdm
import pandas as pd

from tcc.data import CelebALoader

import matplotlib.pyplot as plt

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Plots the data proportions of the CelebA dataset')
    parser.add_argument('--dataset_path',
        required=True,
        type=str,
        help='Path to the CelebA data'
    )
    parser.add_argument('--mirrored',
        action='store_true',
        help='Set if the CelebA loader should consider mirrored data'
    )
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Directory where the generated images will be saved'
    )
    parser.add_argument('--data_size',
        required=False,
        type=int,
        default=60000,
        help='Size of the sub-set of data to consider'
    )

    params = parser.parse_args()

    assert params.data_size >=1, "Parameter '--data_size' must be greater than zero. Received {}".format(params.data_size)

    params.dataset_path = os.path.expandvars(params.dataset_path)
    assert os.path.exists(params.dataset_path), "Data path '{}' does not exist".format(params.dataset_path)
    if not os.path.isabs(params.dataset_path):
        params.dataset_path = os.path.abspath('.' + os.path.sep + params.dataset_path)

    if not os.path.isabs(params.output_folder):
        params.output_folder = os.path.abspath('.' + os.path.sep + params.output_folder)
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    return params

if __name__ == "__main__":
    params = parse()

    loader_params = {
        'batch_size': 1,
        'shuffle': False,
        # 'collate_fn': lambda x: default_collate(x).to(device),
        'num_workers': 0,
        'pin_memory': False
    }

    params.transform = "celebaImagePreprocessTransform"
    params.crop_size = 148
    params.to_image_size = 128
    params.file_extension = ".png"
    params.max_samples = params.data_size
    params.mirror_data = True

    train_gen = CelebALoader(params, "train", **loader_params)
    test_gen = CelebALoader(params, "test", **loader_params)
    eval_gen = CelebALoader(params, "eval", **loader_params)

    dsets = [("Train data", train_gen.dataset.data), ("Test data", test_gen.dataset.data), ("Eval data", eval_gen.dataset.data)]
    
    for (dname, dset) in tqdm(dsets):
        cols = [c for c in dset.columns if c != "image" and c != "mirrored"]
        dset = dset[cols]

        bools = pd.DataFrame()

        bools[cols] = dset == True

        bools.apply(pd.value_counts).T.plot.bar()

        plt.title("{} attribute distribution".format(dname))
        plt.ylabel("#")

        plt.tight_layout()
        plt.savefig(params.output_folder + "/{}-dists".format(dname.split(' ')[0]))
        plt.clf()
