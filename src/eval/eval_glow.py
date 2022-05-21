import os
import sys

from natsort import natsorted
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

from types import SimpleNamespace
import random
import gc

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms

from nflows.transforms.base import MultiscaleCompositeTransform

import matplotlib.pyplot as plt

from tcc.model import Glow
from tcc.eval import ModelLoader
from tcc.util import directory as dir
from tcc.util import loader_functions as lf
from tcc.data import CelebALoader
from tcc.transforms.nflowsTransforms import MultiscaleCompositeTransformAux
from torch.utils.data.dataloader import default_collate
from tcc.util.vector import interpolate
from tcc.util.image import \
    build_horizontal_image_sequence, build_vertical_image_sequence, \
    show_encoded_channels, figure_to_pil_image, \
    add_text_to_PIL_image

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
    parser = argparse.ArgumentParser(description='Generate images from a trained model')
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
    parser.add_argument('--mean_images',
        required=False,
        action='store_true',
        help='Generates the mean image learned by the model'
    )
    parser.add_argument('--sample',
        required=False,
        action='store_true',
        help='Samples from the model'
    )
    parser.add_argument('-n', '--num_samples',
        required=False,
        type=int,
        default=10,
        help='Number of samples to be generated per epoch. Not dependent on parameter \'sample_tampered\''
    )
    parser.add_argument('--sample_tempered',
        required=False,
        action='store_true',
        help='Samples from the model with tempered distribution. Not dependent on parameter \'num_samples\''
    )
    parser.add_argument('--temperature',
        required=False,
        type=float,
        default=0.8,
        help='\"Temperature\" of the tempered sampling'
    )
    parser.add_argument('--interpolate',
        required=False,
        action='store_true',
        help='Interpolates the latent variables of two images generating num_samples samples.'
    )
    parser.add_argument('--show_latents',
        required=False,
        action='store_true',
        help='Generates images for the latent variables of one test image'
    )
    parser.add_argument('-a', '--all_algorithms',
        required=False,
        action='store_true',
        help='Uses all algorithms available'
    )
    parser.add_argument('--zero_scale_latents',
        required=False,
        type=int,
        metavar='scale',
        default=[],
        nargs='+',
        help='Generates an image with the latents of the given scale zeroed. Select multiple scales by providing its IDs separated by spaces. Scale indexing starts at 0.'
    )
    def pair_latent_block_value(s):
        try:
            latBID, value = s.split(',')
            return int(latBID), float(value)
        except:
            raise argparse.ArgumentTypeError("Could not parse '{}' as (id, value).".format(s))

    parser.add_argument('--add_noise_to_scale_latents',
        required=False,
        type=pair_latent_block_value,
        metavar='lat_block_id,noise',
        default=[],
        nargs='+',
        help='Adds uniform noise between [-.5, .5) to a subset of the latent variables acording to pairs of (lat_block_id, r), where \'r\' multiplies the generated noise. Latent block indexing starts at 0. Scale and multiplying factor must be written separated by a comma and with no spaces. Spaces separate the pairs.'
    )
    parser.add_argument('--swap_latents',
        required=False,
        type=int,
        metavar='scale',
        default=[],
        nargs='+',
        help='Generates an image with the latents of the given scale zeroed. Select multiple scales by providing its IDs separated by spaces. Scale indexing starts at 0.'
    )
    parser.add_argument('--sweep_latents',
        required=False,
        type=int,
        metavar='latent_block_id',
        default=[],
        nargs='+',
        help='Generates ten images by sweeping the latents in the interval [v-1, v+1]. The latent block starts at 0 and goes up to the number of scales. The latent block generated after the last scale is divided in two.'
    )
    parser.add_argument('--negate_latents',
        required=False,
        type=int,
        metavar='lat_block_id',
        default=[],
        nargs='+',
        help='Generates images from the negated latent blocks of the original image'
    )
    parser.add_argument('--scale_latents',
        required=False,
        type=pair_latent_block_value,
        metavar='lat_block_id,scale',
        default=[],
        nargs='+',
        help='Generates images from the scaled latent blocks of an original image'
    )
    parser.add_argument('--resample_latents',
        required=False,
        type=pair_latent_block_value,
        metavar='lat_block_id,noise',
        default=[],
        nargs='+',
        help='Generates images after resampling the latent blocks of an original image'
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

    assert params.num_samples > 0, "Number of samples must be greater than zero. Received {}".format(params.num_samples)
    
    if params.all_algorithms:
        params.sample = True
        params.sample_tempered = True
        params.interpolate = True
        params.show_latents = True
        params.mean_images = True

    if params.zero_scale_latents != []:
        params.zero_scale_latents = list(set([int(s) for s in params.zero_scale_latents]))
        params.zero_scale_latents = sorted(params.zero_scale_latents)

    if params.add_noise_to_scale_latents != []:
        params.add_noise_to_scale_latents = [(int(s), float(n)) for s, n in params.add_noise_to_scale_latents]
        params.add_noise_to_scale_latents = sorted(params.add_noise_to_scale_latents, key=lambda x: x[0])

    if params.swap_latents != []:
        params.swap_latents = list(set([int(s) for s in params.swap_latents]))
        params.swap_latents = sorted(params.swap_latents)

    if params.sweep_latents != []:
        params.sweep_latents = list(set([int(s) for s in params.sweep_latents]))
        params.swap_latents = sorted(params.swap_latents)

    if params.negate_latents != []:
        params.negate_latents = list(set([int(s) for s in params.negate_latents]))

    if params.scale_latents != []:
        params.scale_latents = [(int(lid), float(v)) for lid, v in params.scale_latents]
        params.scale_latents = sorted(params.scale_latents, key=lambda x: x[0])

    if params.resample_latents != []:
        params.resample_latents = [(int(lid), float(v)) for lid, v in params.resample_latents]
        params.resample_latents = sorted(params.resample_latents, key=lambda x: x[0])

    algorithm_selected = params.sample \
                         or params.sample_tempered \
                         or params.interpolate \
                         or params.show_latents \
                         or params.mean_images \
                         or len(params.add_noise_to_scale_latents) > 0 \
                         or len(params.zero_scale_latents) > 0 \
                         or len(params.swap_latents) > 0 \
                         or len(params.sweep_latents) > 0 \
                         or len(params.negate_latents) > 0 \
                         or len(params.scale_latents) > 0 \
                         or len(params.resample_latents) > 0

    assert algorithm_selected, "Some algorithm must be selected. For usage help use the '-h' flag."

    params.model_path = models[params.model]

    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

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

    batch, _ = next(iter(test_gen))
    batch = batch.to(device).float()
    imgs = batch[:2]

    mean_rows = []
    sample_img_rows = []
    tempered_sample_img_rows = []
    interpolated_img_rows = []
    latents_rows = []
    zeroed_latents_rows = []
    noisy_latent_rows = []
    swap_latents_rows = []
    sweep_latents_rows = []
    negate_latents_rows = []
    scale_latents_rows = []
    resample_latents_rows = []
    with torch.no_grad():
        for i, (k, model) in enumerate(gmLoader['TrainedFlow']):
            if i != len(gmLoader['TrainedFlow'])-1:
                continue
            print("{}/{} - Epoch {}: ".format(i+1, len(gmLoader['TrainedFlow']), k), end='', flush=True)

            if eval_params.interpolate \
                or eval_params.show_latents \
                or eval_params.zero_scale_latents \
                or eval_params.add_noise_to_scale_latents \
                or eval_params.swap_latents \
                or eval_params.sweep_latents \
                or eval_params.negate_latents \
                or eval_params.scale_latents \
                or eval_params.resample_latents:

                lats, _ = model._transform(imgs)

            if eval_params.mean_images:
                x_rec, ldj = model._transform.inverse(model.mean_().float())
                
                lat_prob = model._distribution._log_prob(model.mean_().float(), None)
                x_prob = lat_prob - ldj

                text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(x_prob.item(), lat_prob.item())
                
                img = transforms.ToPILImage(mode='RGB')(x_rec.squeeze().to(torch.uint8).cpu())
                               
                mean_rows.append(add_text_to_PIL_image(text, img))
                
                del x_rec
                del lat_prob
                del ldj
                del img
                
                print('.', end='', flush=True)

            if eval_params.sample:
                lat_samples, lat_probs = model.get_tempered_latent_samples(eval_params.num_samples, tau=1)
                x_rec, ldj = model._transform.inverse(lat_samples)
                probs = lat_probs - ldj

                x_rec = x_rec.to(torch.uint8).cpu()
                
                images = []
                for j, (im, prob, latprob) in enumerate(zip(x_rec, probs, lat_probs)):
                    img = transforms.ToPILImage(mode='RGB')(im)
                    text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(prob.item(), latprob.item())
                    images.append(add_text_to_PIL_image(text, img))

                sample_img_rows.append(build_horizontal_image_sequence(images))
                
                del x_rec
                del lat_samples
                del lat_probs

                print('.', end='', flush=True)

            if eval_params.sample_tempered:
                lat_samples, lat_probs = model.get_tempered_latent_samples(n=eval_params.num_samples, tau=eval_params.temperature)
                x_rec, ldj = model._transform.inverse(lat_samples)
                probs = lat_probs - ldj

                x_rec = x_rec.to(torch.uint8).cpu()
                
                images = [] 
                for j, (im, prob, latprob) in enumerate(zip(x_rec, probs, lat_probs)):
                    img = transforms.ToPILImage(mode='RGB')(im)
                    text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(prob.item(), latprob.item())
                    images.append(add_text_to_PIL_image(text, img))

                tempered_sample_img_rows.append(build_horizontal_image_sequence(images))
            
                del x_rec
                del lat_probs
                del lat_samples
                del images

                print('.', end='', flush=True)

            if eval_params.interpolate:
                interpolated_latents = interpolate(lats[0], lats[1], eval_params.num_samples)

                lat_probs = model._distribution._log_prob(interpolated_latents, None)

                x_rec, ldj = model._transform.inverse(interpolated_latents)
                x_probs = lat_probs - ldj
                
                x_rec = x_rec.to(torch.uint8).cpu()

                x_orig = imgs[:2]
                lat_orig, ldj = model._transform(x_orig)
                lat_orig_prob = model._distribution._log_prob(lat_orig, None)
                x_orig_prob = lat_orig_prob + ldj

                text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(x_orig_prob[0].item(), lat_orig_prob[0].item())
                img = add_text_to_PIL_image(text, transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8)))

                images = [img]

                for j, (im, prob, latprob) in enumerate(zip(x_rec, probs, lat_probs)):
                    img = transforms.ToPILImage(mode='RGB')(im)
                    text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(prob.item(), latprob.item())
                    images.append(add_text_to_PIL_image(text, img))
    
                text = "ln p(x)={:.2E}\nln p(u)={:.2E}".format(x_orig_prob[1].item(), lat_orig_prob[1].item())
                img = add_text_to_PIL_image(text, transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8)))
                images += [img]

                interpolated_images = build_horizontal_image_sequence(images)
                interpolated_img_rows.append(interpolated_images)

                del images
                del interpolated_latents
                del interpolated_images #
                del x_rec

                print('.', end='', flush=True)

            if eval_params.show_latents:
                lats_shape = lats.shape
                lats = lats.reshape(imgs.shape)
                fig = show_encoded_channels(lats[0].cpu().numpy(), True)
                img = figure_to_pil_image(fig)
                plt.close(fig)

                latents_rows.append(img)
                
                del img

                print('.', end='', flush=True)

            if eval_params.zero_scale_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:                        
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s in eval_params.zero_scale_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when zeroing latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] = 0

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        zeroed_latents_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.add_noise_to_scale_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c
                        # print(split_indices) # TODO check indices of model

                        for s, n in eval_params.add_noise_to_scale_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when adding noise to latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] += (torch.rand(index1_-index0_, device=device) - 0.5) * n

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        noisy_latent_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.swap_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s in eval_params.swap_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when swaping latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            tmp = torch.clone(lats_[0, split_indices[s] : split_indices[s + 1]])
                            lats_[0, index0_ : index1_] = lats_[1, index0_ : index1_]
                            lats_[1, index0_ : index1_] = tmp

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        swap_latents_rows.append(build_horizontal_image_sequence(images))

                        del tmp
                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.sweep_latents != []:
                sweep_bottom = -1
                sweep_top = 1
                n_steps = 10
                step_size = (sweep_top - sweep_bottom)/(n_steps-1)
                sweep_values = torch.arange(sweep_bottom, sweep_top+1e-5, step_size, device=device)
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        # print("\n",sweep_values)
                        lats_ = torch.clone(lats[0])
                        # print(lats_[:30])
                        lats_ = lats_.repeat(*sweep_values.shape, 1)
                        # print(lats_[:, :30])

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s in eval_params.sweep_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when sweeping through latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] = (torch.repeat_interleave(
                                    lats_[0, index0_ : index1_].reshape(-1,1), 
                                    repeats=sweep_values.shape[-1], 
                                    dim=1
                                    ) + sweep_values
                                ).T.view(-1, *lats_[0, index0_ : index1_].shape)

                        # print(lats_[:, :30])
                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 

                        sweep_latents_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.negate_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s in eval_params.negate_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when negating latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] = -lats_[:, index0_ : index1_]

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        negate_latents_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.scale_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s, n in eval_params.scale_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when scaling latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] = lats_[:, index0_ : index1_]*n

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        scale_latents_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)

            if eval_params.resample_latents != []:
                for i, t in enumerate(model._transform._transforms):
                    if type(t) == MultiscaleCompositeTransform:
                        lats_ = torch.clone(lats)

                        c = t.__class__
                        t.__class__ = MultiscaleCompositeTransformAux
                        split_indices = t.get_split_indices()
                        t.__class__ = c

                        for s, n in eval_params.resample_latents:
                            if s > t._num_transforms:
                                print("\t- Ignoring invalid scale ID '{}' when resampling latents".format(s))
                                continue
                            elif s == t._num_transforms:
                                index0_ = split_indices[s] - (split_indices[s]-split_indices[s-1])//2
                                index1_ = split_indices[s]
                            elif s == t._num_transforms-1:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s] + (split_indices[s+1]-split_indices[s])//2
                            else:
                                index0_ = split_indices[s]
                                index1_ = split_indices[s + 1]

                            lats_[:, index0_ : index1_] = n*torch.randn(*lats_[:, index0_ : index1_].shape)

                        x_t, _ = t.inverse(lats_)
                        invs = [t_.inverse for t_ in model._transform._transforms[:i]]
                        for t_ in reversed(invs):
                            x_t, _ = t_(x_t)

                        x_t = x_t.to(torch.int8).cpu()

                        images = [transforms.ToPILImage(mode='RGB')(imgs[0].to(torch.uint8))]
                        images += [transforms.ToPILImage(mode='RGB')(x) for x in x_t] 
                        images += [transforms.ToPILImage(mode='RGB')(imgs[1].to(torch.uint8))]
                        
                        resample_latents_rows.append(build_horizontal_image_sequence(images))

                        del lats_
                        del x_t
                        del images
                        
                print('.', end='', flush=True)


            del model
            gc.collect()

            print()


    print('Done computing. Saving...', flush=True)

    if eval_params.mean_images:
        fname = "{}/mean_imgs.png".format(eval_params.output_folder)
        mean_imgs = build_vertical_image_sequence(mean_rows)
        mean_imgs.save(fname, "PNG")
        print("Saved mean images to {}".format(fname))
        
    if eval_params.sample:
        fname = "{}/sample_imgs.png".format(eval_params.output_folder)
        sample_imgs = build_vertical_image_sequence(sample_img_rows)
        sample_imgs.save(fname, "PNG")
        print("Saved sample images to {}".format(fname))
    
    if eval_params.sample_tempered:
        fname = "{}/tempered_sample_imgs-temp{}.png".format(eval_params.output_folder, eval_params.temperature)
        tempered_sample_imgs = build_vertical_image_sequence(tempered_sample_img_rows)
        tempered_sample_imgs.save(fname, "PNG")
        print("Saved tempered sample images to {}".format(fname))
    
    if eval_params.interpolate:
        fname = "{}/interpolated_imgs.png".format(eval_params.output_folder)
        interpolated_imgs = build_vertical_image_sequence(interpolated_img_rows)
        interpolated_imgs.save(fname, "PNG")
        print("Saved interpolated images to {}".format(fname))

    if eval_params.show_latents:
        fname = "{}/latent_imgs.png".format(eval_params.output_folder)
        latent_imgs = build_vertical_image_sequence(latents_rows)
        latent_imgs.save(fname, "PNG")
        print("Saved latent images to {}".format(fname))

    if eval_params.zero_scale_latents != []:
        suffix_str = ''.join(['s{}'.format(s) for s in eval_params.zero_scale_latents])
        fname = "{}/zeroed_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(zeroed_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated with noisy latents at {}".format(fname))

    if eval_params.add_noise_to_scale_latents != []:
        suffix_str = ''.join(['s{}n{}'.format(s,n) for s, n in eval_params.add_noise_to_scale_latents])
        fname = "{}/noisy_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(noisy_latent_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated with noisy latents at {}".format(fname))
    
    if eval_params.swap_latents != []:
        suffix_str = ''.join(['s{}'.format(s) for s in eval_params.swap_latents])
        fname = "{}/swaped_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(swap_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated with noisy latents at {}".format(fname))

    if eval_params.sweep_latents != []:
        suffix_str = ''.join(['s{}'.format(s) for s in eval_params.sweep_latents])
        fname = "{}/sweeped_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(sweep_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated from sweeping values of latents at {}".format(fname))
        
    if eval_params.negate_latents != []:
        suffix_str = ''.join(['lbid{}'.format(s) for s in eval_params.negate_latents])
        fname = "{}/negated_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(negate_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated from negating latents at {}".format(fname))

    if eval_params.scale_latents != []:
        suffix_str = ''.join(['lbid{}s{}'.format(s,n) for s, n in eval_params.scale_latents])
        fname = "{}/scaled_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(scale_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated with scaled latents at {}".format(fname))

    if eval_params.resample_latents != []:
        suffix_str = ''.join(['lbid{}n{}'.format(s,n) for s, n in eval_params.resample_latents])
        fname = "{}/resampled_latents-{}.png".format(eval_params.output_folder, suffix_str)
        partial_transform_inverse_imgs = build_vertical_image_sequence(resample_latents_rows)
        partial_transform_inverse_imgs.save(fname, "PNG")
        print("Saved images generated with resampled latents at {}".format(fname))

    print("Done.")
