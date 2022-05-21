import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

import numpy as np
import torch
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from types import SimpleNamespace
from joblib import dump, load
from torchvision import transforms

from tcc.data import CelebALoader
from tcc.eval import ModelLoader
from tcc.util import directory as dir
from tcc.util import loader_functions as lf
from tcc.util.image import build_horizontal_image_sequence, build_vertical_image_sequence, add_text_to_PIL_image

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Plot SVM related results, such as accuracies and image manipulations')
    parser.add_argument('--celeba_path',
        required=False,
        type=str,
        default='',
        help='Path to the CelebA data'
    )
    parser.add_argument('--glow_model_parent_dir',
        required=False,
        type=str,
        default='',
        help='Path to the trained Glow model'
    )
    parser.add_argument('--svm_path',
        required=False,
        type=str,
        default='',
        help='Path to the SVM model or parent directory. See \'is_folder\''
    )
    parser.add_argument('--is_folder',
        required=False,
        action='store_true',
        help='If set, generate results for all SVM models saved as .joblib under the directory pointed by \'svm_path\''
    )
    parser.add_argument('--generate_images',
        required=False,
        action='store_true',
        help='If set, will generate a number of images based on the SVM(s) hyperplanes'
    )
    parser.add_argument('--plot_accuracies',
        required=False,
        action='store_true',
        help='If set, the accuracies pointed by \'acc_csv\' will be plotted'
    )
    parser.add_argument('--acc_csv',
        required=False,
        type=str,
        default='',
        help='The path to the CSV containing the accuracies of all SVMs'
    )
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Directory where the generated images will be saved'
    )
    parser.add_argument('--num_obs',
        required=False,
        type=int,
        default=1,
        help='Number of observations to use for the sample generation. Max. 50.'
    )
    parser.add_argument('--fname_suffix',
        required=False,
        type=str,
        default=None,
        help='Suffix to be added to the end of file names of the generated files. Will not be filtered'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )

    params = parser.parse_args()

    assert params.num_obs >=1 and params.num_obs <= 50, "Parameter '--num_obs' must be in the interval 1 <= n <= 50. Received {}".format(params.num_obs)
    
    if params.celeba_path != '':
        params.celeba_path = os.path.expandvars(params.celeba_path)
        assert os.path.exists(params.celeba_path), "Data path '{}' does not exist".format(params.celeba_path)
        if not os.path.isabs(params.celeba_path):
            params.celeba_path = os.path.abspath('.' + os.path.sep + params.celeba_path)

    if params.celeba_path != '':
        params.glow_model_parent_dir = os.path.expandvars(params.glow_model_parent_dir)
        assert os.path.exists(params.glow_model_parent_dir), "Data path '{}' does not exist".format(params.glow_model_parent_dir)
        if not os.path.isabs(params.glow_model_parent_dir):
            params.glow_model_parent_dir = os.path.abspath('.' + os.path.sep + params.glow_model_parent_dir)

    if params.svm_path != '':   
        params.svm_path = os.path.expandvars(params.svm_path)
        assert os.path.exists(params.svm_path), "Data path '{}' does not exist".format(params.svm_path)
        if not os.path.isabs(params.svm_path):
            params.svm_path = os.path.abspath('.' + os.path.sep + params.svm_path)

    if params.acc_csv != '':
        params.acc_csv = os.path.expandvars(params.acc_csv)
        assert os.path.exists(params.acc_csv), "Data path '{}' does not exist".format(params.acc_csv)
        if not os.path.isabs(params.acc_csv):
            params.acc_csv = os.path.abspath('.' + os.path.sep + params.acc_csv)

    if params.is_folder and params.svm_path != '':
        params.svm_path = dir.list_dir(params.svm_path, dir.ListDirPolicyEnum.FILES_ONLY)
    else:
        params.svm_path = [params.svm_path]

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

    if params.generate_images:

        with torch.no_grad():

            def load_glow(path, device=None):
                return torch.load(path, map_location=device)

            device = torch.device('cpu')
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            loaders_dict = {
                'Params': lf.json_load_fn,
                'TrainLosses': lf.torch_tensor_load_fn,
                'TestLosses': lf.torch_tensor_load_fn,
                'TrainedFlow': lambda x: load_glow(x, device=device),
            }
            gmLoader = ModelLoader(params.glow_model_parent_dir, loaders_dict=loaders_dict)

            model_params = gmLoader['Params'][0]
            for _, p in model_params.iterrows():
                print(p)

            model_params = SimpleNamespace(**model_params.iloc[0].to_dict())
            model_params.dataset_path = "{}{}..{}..{}data{}".format(FPATH, *[os.sep]*4) + model_params.dataset_path.split('data'+os.sep)[-1] 

            model = gmLoader['TrainedFlow'][len(gmLoader['TrainedFlow'])-1]

            loader_params = {
                'batch_size': model_params.batch_size,
                'shuffle': False,
                # 'collate_fn': lambda x: default_collate(x).to(device),
                'num_workers': 0,
                'pin_memory': False
            }

            test_gen = CelebALoader(model_params, "eval", **loader_params)
            
            batch, _ = next(iter(test_gen))
            batch = batch.to(device).float()
            imgs = batch[:params.num_obs]
            
            lats, orig_logabsdet = model._transform(imgs)
            
            modifiers = torch.linspace(-20, 20, 20).float()

            for i, svmpath in enumerate(params.svm_path):
                svm_name = svmpath.split('/')[-1].split('.')[0]
                print("{}/{} Generating image for SVM '{}'".format(i+1, len(params.svm_path), svm_name))
                clf = load(svmpath)

                print('W vector magnitude:', np.linalg.norm(clf.coef_))
                w = clf.coef_ / np.linalg.norm(clf.coef_)
                b = clf.intercept_

                lat_probs = model._distribution._log_prob(lats, None).cpu()
                probs = lat_probs + orig_logabsdet
                res_images = []
                for im, prob, latprob in zip(imgs, probs, lat_probs):
                    img = transforms.ToPILImage(mode='RGB')(im.to(torch.uint8))
                                            
                    text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(prob.item(), latprob.item())

                    res_images.append([add_text_to_PIL_image(text, img)])

                for m in tqdm(modifiers):
                    lats_mod = (lats + m * w).float()
                    x_rec, logabsdet = model._transform.inverse(lats_mod)
                    lat_probs = model._distribution._log_prob(lats_mod, None).cpu()
                    probs = lat_probs - logabsdet
                    
                    x_rec = x_rec.to(torch.uint8).cpu()
                    for j, (im, prob, latprob) in enumerate(zip(x_rec, probs, lat_probs)):
                        img = transforms.ToPILImage(mode='RGB')(im)
                                            
                        text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(prob.item(), latprob.item())
                        

                        res_images[j].append(add_text_to_PIL_image(text, img))
                        
                
                for j, hor_img in enumerate(res_images):
                    res_images[j] = build_horizontal_image_sequence(hor_img)
                
                res_image = build_vertical_image_sequence(res_images)

                fname = "{}/{}-resulting_images-numObs{}{}.png".format(params.output_folder, svm_name, params.num_obs, (('-' + params.fname_suffix) if params.fname_suffix else ''))
                res_image.save(fname, "PNG")
                print("Saved resulting images to {}".format(fname))

    if params.plot_accuracies:
        df = pd.read_csv(params.acc_csv)

        fig, ax = plt.subplots()

        df.sort_values(by='accuracy', inplace=True)

        pd.DataFrame({'accuracy': df['accuracy'].values}, index=df['name']).plot.bar(ax=ax)
        
        ax.set_xlabel(None)
        plt.tight_layout()
        fig.savefig(params.output_folder + '/svm_accuracies')
        
