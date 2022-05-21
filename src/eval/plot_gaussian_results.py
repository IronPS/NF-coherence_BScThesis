
import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

from tqdm import tqdm

import numpy as np
import torch
import random

import pandas as pd
import matplotlib.pyplot as plt

from types import SimpleNamespace
from joblib import dump, load
from torchvision import transforms

from tcc.distribution import NormalGamma, DiagonalGaussian
from tcc.eval import ModelLoader
from tcc.util import directory as dir
from tcc.util import loader_functions as lf
from tcc.util.image import build_horizontal_image_sequence, build_vertical_image_sequence, add_text_to_PIL_image


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Generates multiple plots for the class attribute Gaussian distributions')
    parser.add_argument('--celeba_path',
        required=True,
        type=str,
        help='Path to the CelebA data'
    )
    parser.add_argument('--glow_model_parent_dir',
        required=True,
        type=str,
        help='Path to the trained Glow model'
    )
    parser.add_argument('--gaussian_path',
        required=False,
        type=str,
        help='Path to the Gaussian model or parent directory. If \'is_folder\' is set, this must be a folder with only .joblib files'
    )
    parser.add_argument('--lprobs_path',
        required=False,
        type=str,
        default='',
        help='The CSV file of the already calculated log-likelihoods of both positively and negatively labeled data. If \'is_folder\' is set, this must be a folder with only CSV files.'
    )
    parser.add_argument('--is_folder',
        required=False,
        action='store_true',
        help='If set, will interpred \'gaussia_path\' and \'lprobs_path\' as directories'
    )
    parser.add_argument('--plot_lprobs',
        action='store_true',
        help='Plots the data in the CSV file(s) pointed by \'lprobs_path\''
    )
    parser.add_argument('--lprobs_confusion',
        action='store_true',
        help='Uses the CSV data to calculate a confusion matrix using the maximum likelihood probability as a means of decision'
    )
    parser.add_argument('--output_folder',
        required=False,
        type=str,
        default='.',
        help='Directory where the generated images will be saved'
    )
    parser.add_argument('--mean',
        action='store_true',
        help='If set, will generate the mean images of the Gaussian cluster'
    )
    parser.add_argument('--mean_vlines',
        action='store_true',
        help='If set, will generate a single plot with the likelihood of the means plotted as vertical lines'
    )
    parser.add_argument('--sample', 
        action='store_true',
        help='If set,  will sample \'num_samples\' samples from the Gaussian model'
    )
    parser.add_argument('--num_samples',
        required=False,
        type=int,
        default=1,
        help='Number of observations to use for the sample generation. Max. 50.'
    )
    parser.add_argument('--temperature',
        required=False,
        type=float,
        default=0.5,
        help='The temperature used effectively scales the noise. 0 < parameter'
    )
    parser.add_argument('--seed',
        required=False,
        type=int,
        default=0,
        help='Sets a random seed for the result generation'
    )

    params = parser.parse_args()

    assert params.num_samples >=1 and params.num_samples <= 50, "Parameter '--num_samples' must be in the interval 1 <= n <= 50. Received {}".format(params.num_samples)
    assert params.temperature > 0, "Parameter '--temperature' must be positive. Received {}".format(params.temperature)

    params.celeba_path = os.path.expandvars(params.celeba_path)
    assert os.path.exists(params.celeba_path), "Data path '{}' does not exist".format(params.celeba_path)
    if not os.path.isabs(params.celeba_path):
        params.celeba_path = os.path.abspath('.' + os.path.sep + params.celeba_path)

    params.glow_model_parent_dir = os.path.expandvars(params.glow_model_parent_dir)
    assert os.path.exists(params.glow_model_parent_dir), "Data path '{}' does not exist".format(params.glow_model_parent_dir)
    if not os.path.isabs(params.glow_model_parent_dir):
        params.glow_model_parent_dir = os.path.abspath('.' + os.path.sep + params.glow_model_parent_dir)
    
    params.gaussian_path = os.path.expandvars(params.gaussian_path)
    assert os.path.exists(params.gaussian_path), "Data path '{}' does not exist".format(params.gaussian_path)
    if not os.path.isabs(params.gaussian_path):
        params.gaussian_path = os.path.abspath('.' + os.path.sep + params.gaussian_path)

    if params.plot_lprobs or params.lprobs_confusion:
        params.lprobs_path = os.path.expandvars(params.lprobs_path)
        assert os.path.exists(params.lprobs_path), "Data path '{}' does not exist".format(params.lprobs_path)
        if not os.path.isabs(params.lprobs_path):
            params.lprobs_path = os.path.abspath('.' + os.path.sep + params.lprobs_path)

    if params.is_folder:
        params.gaussian_path = dir.list_dir(params.gaussian_path, dir.ListDirPolicyEnum.FILES_ONLY)

        if params.plot_lprobs or params.lprobs_confusion:
            params.lprobs_path = dir.list_dir(params.lprobs_path, dir.ListDirPolicyEnum.FILES_ONLY)
    else:
        params.gaussian_path = [params.gaussian_path]

        if params.plot_lprobs or params.lprobs_confusion:
            params.lprobs_path = [params.lprobs_path]

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

        hist_nbins = 100
        if params.plot_lprobs:
            print("Producing log-probabilities plots...")
            for path in tqdm(params.lprobs_path):
                attr_name = '-'.join(path.split('/')[-1].split('-')[:-1])
                df = pd.read_csv(path)
                

                df_p = df['p_lprobs']
                
                df_p_true = df_p.loc[df['positiveAttr'] == True]
                df_p_false = df_p.loc[df['positiveAttr'] != True]

                
                fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

                fig.suptitle("{} - Log-likelihood attributions".format(attr_name))

                axes[0].hist(df_p_true, bins=hist_nbins, color='blue', alpha=0.5, label='True Positives')
                axes[0].hist(df_p_false, bins=hist_nbins, color='red', alpha=0.5, label='True Negatives')
                
                axes[0].set_title("Positive Label Distribution")
                axes[0].set_ylabel("#")
                axes[0].legend()
                
                df_n = df['n_lprobs']
                df_n_true = df_n.loc[df['positiveAttr'] == True]
                df_n_false = df_n.loc[df['positiveAttr'] != True]

                axes[1].hist(df_n_true, bins=hist_nbins, color='blue', alpha=0.5, label='True Positive')
                axes[1].hist(df_n_false, bins=hist_nbins, color='red', alpha=0.5, label='True Negatives')
                
                axes[1].set_title("Negative Label Distribution")
                axes[1].set_xlabel("Log-likelihood")
                axes[1].set_ylabel("#")
                axes[1].legend()

                n_true_hist = np.histogram(df_n_true, bins='auto')
                n_false_hist = np.histogram(df_n_false, bins='auto')
                
                figpath = params.output_folder + '/lprobs_hist-{}.png'.format(attr_name)
                print("Saving to {}".format(figpath))
                fig.savefig(figpath)
                plt.close(fig)

        if params.lprobs_confusion:
            print("Producing confusion matrices...")
            for path in tqdm(params.lprobs_path):
                attr_name = '-'.join(path.split('/')[-1].split('-')[:-1])
                df = pd.read_csv(path)

                N = df.shape[0]
                N_p = np.count_nonzero(df['positiveAttr'] == True)
                N_n = np.count_nonzero(df['positiveAttr'] != True)
                
                df_decisions = df['p_lprobs'] > df['n_lprobs']

                true_positives = (df_decisions) & (df['positiveAttr'] == True)
                false_positives = (df_decisions) & (df['positiveAttr'] != True)

                true_negatives = (~df_decisions) & (df['positiveAttr'] != True)
                false_negatives = (~df_decisions) & (df['positiveAttr'] == True)

                n_true_positives = np.count_nonzero(true_positives)
                n_false_positives = np.count_nonzero(false_positives)
                n_true_negatives = np.count_nonzero(true_negatives)
                n_false_negatives = np.count_nonzero(false_negatives)
               
                confusion_mat = np.asarray([[n_false_positives, n_true_negatives], [n_true_positives, n_false_negatives]]) / N


                figpath = params.output_folder + '/lprobs_confusion-{}.png'.format(attr_name)

                heatmap = plt.pcolor(confusion_mat)#, vmin=0, vmax=1)
                plt.title("{} - Confusion Matrix".format(attr_name))
                plt.text(0.65, 0.02, "{:<17} {:5.2f}\n{:<17} {:5.2f}".format("% Positive labels", 100*N_p/N, "% Negative labels", 100*N_n/N), family='monospace', transform=plt.gcf().transFigure)

                for x in range(confusion_mat.shape[0]):
                    for y in range(confusion_mat.shape[1]):
                        plt.text(x + 0.5, y + 0.5, "{:.2f}%".format(confusion_mat[y, x]*100),
                                horizontalalignment='center',
                                verticalalignment='center',
                                color='black'
                                )

                plt.colorbar(heatmap)

                plt.xlabel("Prediction")
                plt.xticks([0.5, 1.5], ["T", "F"])
                plt.ylabel("True Value")
                plt.yticks([0.5, 1.5], ["F", "T"])
                plt.savefig(figpath)
                plt.clf()

        if params.mean_vlines:
            print("Generating mean vertical lines plot...")
            fig_p, axes_p = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(10, 6))
            fig_n, axes_n = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(10, 6))

            min_x_p, max_x_p = np.inf, -np.inf
            min_l_p, max_l_p = np.inf, -np.inf
            min_d_p, max_d_p = np.inf, -np.inf
            min_x_n, max_x_n = np.inf, -np.inf
            min_l_n, max_l_n = np.inf, -np.inf
            min_d_n, max_d_n = np.inf, -np.inf

            for path in tqdm(params.gaussian_path):
                letter, name = path.split('/')[-1].split('.')[0].split('-')
                name = '_'.join(name.split('_')[:-1])

                ng_model = load(path)

                pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision()))

                lat = ng_model.mean_mean().unsqueeze(0)

                dist_prob = pred_dg_model.lpdf(lat)
                lat_prob = model._distribution._log_prob(lat, None)

                x_rec, ldj = model._transform.inverse(lat)
                x_prob = lat_prob - ldj

                positive = letter == 'p'
                if positive:
                    axes = axes_p
                    max_x_p = x_prob.item() if x_prob.item() > max_x_p else max_x_p
                    min_x_p = x_prob.item() if x_prob.item() < min_x_p else min_x_p
                    max_l_p = lat_prob.item() if lat_prob.item() > max_l_p else max_l_p
                    min_l_p = lat_prob.item() if lat_prob.item() < min_l_p else min_l_p
                    max_d_p = dist_prob.item() if dist_prob.item() > max_d_p else max_d_p
                    min_d_p = dist_prob.item() if dist_prob.item() < min_d_p else min_d_p

                else:
                    axes = axes_n
                    max_x_n = x_prob.item() if x_prob.item() > max_x_n else max_x_n
                    min_x_n = x_prob.item() if x_prob.item() < min_x_n else min_x_n
                    max_l_n = lat_prob.item() if lat_prob.item() > max_l_n else max_l_n
                    min_l_n = lat_prob.item() if lat_prob.item() < min_l_n else min_l_n
                    max_d_n = dist_prob.item() if dist_prob.item() > max_d_n else max_d_n
                    min_d_n = dist_prob.item() if dist_prob.item() < min_d_n else min_d_n

                trans_dist = axes[0].get_xaxis_transform()
                trans_lat = axes[1].get_xaxis_transform()
                trans_x = axes[2].get_xaxis_transform()

                axes[0].axvline(dist_prob.item(), alpha=0.3, color='blue')
                axes[0].text(dist_prob.item(), 1, name, transform=trans_dist, rotation=-60, horizontalalignment='right', verticalalignment='bottom')
                axes[1].axvline(lat_prob.item(), alpha=0.3, color='blue')
                axes[1].text(lat_prob.item(), 1, name, transform=trans_lat, rotation=-60, horizontalalignment='right', verticalalignment='bottom')
                axes[2].axvline(x_prob.item(), alpha=0.3, color='blue')
                axes[2].text(x_prob.item(), 1, name, transform=trans_x, rotation=-60, horizontalalignment='right', verticalalignment='bottom')

            p_vals = (min_d_p, max_d_p, min_x_p, max_x_p, min_l_p, max_l_p)
            n_vals = (min_d_n, max_d_n, min_x_n, max_x_n, min_l_n, max_l_n)
            for int_, axes in zip([p_vals, n_vals], [axes_p, axes_n]):
                try:
                    min_d, max_d, min_x, max_x, min_l, max_l = int_
                    axes[2].set_xlabel("Log probability")
                
                    int_d = max_d - min_d
                    int_x = max_x - min_x
                    int_l = max_l - min_l

                    print(int_)

                    axes[0].set_xlim(min_d - int_d*0.051, max_d + int_d*0.051 )
                    axes[1].set_xlim(min_l - int_l*0.051, max_l + int_l*0.051 )
                    axes[2].set_xlim(min_x - int_x*0.051, max_x + int_x*0.051 )

                    axes[0].set_yticks([])
                    axes[1].set_yticks([])
                    axes[2].set_yticks([])

                    plt.setp(axes[0].get_yticklabels(), visible=False)
                    plt.setp(axes[1].get_yticklabels(), visible=False)
                    plt.setp(axes[2].get_yticklabels(), visible=False)
                    
                except:
                    pass


            fig_p.tight_layout()
            fig_n.tight_layout()
            fig_p.savefig(params.output_folder + '/p-dists-mean-lprob')
            fig_n.savefig(params.output_folder + '/n-dists-mean-lprob')

            plt.close(fig_p)
            plt.close(fig_n)
            plt.clf()

        if params.mean:
            print("Generating mean images...")
            for path in tqdm(params.gaussian_path):
                name = path.split('/')[-1].split('.')[0]
                ng_model = load(path)
                pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision()))
                
                lat = ng_model.mean_mean().unsqueeze(0)

                dist_prob = pred_dg_model.lpdf(lat)
                lat_prob = model._distribution._log_prob(lat, None)

                x_rec, ldj = model._transform.inverse(lat)
                x_prob = lat_prob - ldj
                                   
                img = transforms.ToPILImage(mode='RGB')(x_rec.squeeze(0).to(torch.uint8).cpu())
                                        
                text = "ln p(x)={:.2E}\nln p(u)={:.2E}\nln p(c)={:.2E}".format(x_prob.item(), lat_prob.item(), dist_prob.item())
                
                img = add_text_to_PIL_image(text, img)

                fname = params.output_folder + "/mean-{}.png".format(name)
                print("Saving mean image to {}".format(fname))
                img.save(fname, "PNG")

                del x_rec
                del lat
                del lat_prob
                del x_prob
                del img

        if params.sample:
            print("Producing samples...")
            for path in tqdm(params.gaussian_path):
                name = path.split('/')[-1].split('.')[0]
                ng_model = load(path)

                pred_dg_model = DiagonalGaussian(ng_model.mean_mean(), 1./torch.sqrt(ng_model.mode_precision()))
            
                samples, prob_probs = pred_dg_model.sample(n=params.num_samples, tau=params.temperature)

                lat_probs = model._distribution._log_prob(samples, None)

                x_rec, ldj = model._transform.inverse(samples)
                x_probs = lat_probs - ldj

                x_rec = x_rec.to(torch.uint8).cpu()

                res_images = []
                for j, (im, prob, latprob, inprob) in enumerate(zip(x_rec, x_probs, lat_probs, prob_probs)):
                    img = transforms.ToPILImage(mode='RGB')(im)
                                        
                    text = "ln p(x)={:.2E}\nln p(u)={:.2E}\nln p(c)={:.2E}".format(prob.item(), latprob.item(), inprob.item())

                    res_images.append(add_text_to_PIL_image(text, img))

                img = build_horizontal_image_sequence(res_images)

                fname = "{}/{}-samples-temp{}.png".format(params.output_folder, name, params.temperature)
                print("Saving sample images to {}".format(fname))
                img.save(fname, "PNG")

                del samples
                del x_rec
                del lat_probs
                del x_probs
                del res_images
                del img
