
import sys
import os
import json
from functools import reduce as Reduce
FPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(FPATH + '/../'))

import torch
from torchvision import transforms
import random
import numpy as np

from tcc.transforms import Dequantization
from tcc.util.image import build_horizontal_image_sequence

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train a Glow model')
    parser.add_argument('--session_name',
        required=True,
        type=str,
        help='Name of the training session. Used to save the trained network'
    )
    parser.add_argument('--dataset_path',
        type=str,
        required=True,
        help='The path to the dataset folder'
    )
    parser.add_argument('--dataset_loader',
        required=True,
        choices=['celeba'],
        default='celeba',
        help='The dataset loader to use'
    )
    parser.add_argument('-m', '--max_samples',
        type=int,
        default=0,
        help='(default 0: whole dataset) The maximum number of samples to be used during training'
    )
    parser.add_argument('--file_extension',
        choices=['pt', 'jpg', 'png'],
        default='jpg',
        help='Extension of the files to be loaded'
    )
    parser.add_argument('--mirror_data',
        action='store_true',
        help='If set, the data images will be duplicated and mirrored'
    )
    parser.add_argument('--save_path',
        required=True,
        type=str,
        help='Path where the training data will be saved'
    )
    parser.add_argument('-t', '--transform',
        choices=['None', 'celebaImagePreprocessTransform'],
        default='None',
        help='The preprocessing function to apply'
    )
    parser.add_argument('--crop_size',
        type=int,
        default=148,
        help='Image preprocessing crop size before resizing'
    )
    parser.add_argument('--to_image_size',
        type=int,
        default=64,
        help='Final size of the image after preprocessing'
    )
    parser.add_argument('-b', '--batch_size',
        type=int,
        default=16,
        help='Number of sample images per step'
    )
    parser.add_argument('-e', '--max_epochs',
        type=int,
        default=50,
        help='Maximum number of epochs in training'
    )
    parser.add_argument('-n', '--num_threads',
        type=int,
        default=0,
        help='Number of working threads for data loading'
    )
    parser.add_argument('-s', '--num_scales', 
        type=int,
        default=3,
        help='Number of scales used'
    )
    parser.add_argument('-l', '--num_flows',
        type=int,
        default=32,
        help='Number of flows contained in a scale'
    )
    parser.add_argument('--conv_mid_channel_num',
        type=int,
        default=128,
        help='Number of channels in the middle layer of the CNN being used'
    )
    parser.add_argument('--base_distribution',
        choices=['std_normal', 'diagonal_normal'],
        default='std_normal',
        help='Base distribution of the Glow model'
    )
    parser.add_argument('--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate used by the AdamW optimizer'
    )
    parser.add_argument('--checkpoints_to_lr_decrease',
        required=False,
        type=int,
        default=24,
        help='The initial number of checkpoints with no loss decrease to trigger the learning rate scheduler. (param > 0)'
    )
    parser.add_argument('--lr_checkpoint_increase_factor',
        required=False,
        type=float,
        default=1.3,
        help='Every learning rate scheduler trigger will determine the next floor(checkpoints_to_lr_decrease *= lr_checkpoint_increase_factor). (param >= 1)'
    )
    parser.add_argument('--lr_decrease_factor',
        required=False,
        type=float,
        default=8e-1,
        help='The learning rate will be modifed according to lr := lr * lr_decrease_factor. 0 < lr_decrease_factor <= 1'
    )
    parser.add_argument('--min_lr',
        required=False,
        type=float,
        default=1e-5,
        help='Minimum learning rate. (param > 0)'
    )
    parser.add_argument('--seed',
        type=int,
        default=0,
        help='Execution seed'
    )
    parser.add_argument('--use_cuda',
        action='store_true'
    )
    # parser.add_argument('--device_id',
    #     type=int,
    #     default=0,
    #     help='The ID of the device, if use_cuda is active'
    # )


    params = parser.parse_args()

    assert params.batch_size >= 1, "Batch size must be greater than zero. Received {}".format(params.batch_size)
    assert params.max_epochs >= 1, "Maximum number of epochs must be greater than zero. Received {}".format(params.max_epochs)
    assert params.max_samples >= 0, "Maximum number of samples must be non-negative. Received " + params.max_samples
    assert params.crop_size >= 1, "Crop size must be greater than zero. Received {}".format(params.crop_siz)
    assert params.to_image_size >= 1, "Resulting image size must be greater than zero. Received {}".format( params.to_image_siz)
    assert params.num_threads >= 0, "Number of threads must be non-negative. Received {}".format(params.num_threads)
    assert params.num_scales >= 1, "Number of scales must be greater than zero. Received {}".format(params.num_scales)
    assert params.num_flows >= 1, "Number of flows per scale must be greater than zero. Received {}".format(params.num_flows)
    assert params.conv_mid_channel_num >= 1, "Number of CNN channels must be greater than zero. Received {}".format(params.conv_mid_channel_num)
    assert params.learning_rate > 0, "Learning rate must be greater than zero. Received {}".format(params.learning_rate)
    assert params.checkpoints_to_lr_decrease > 0 , "Parameter \'checkpoints_to_lr_decrease\' must be greater than zero. Received {}".format(params.checkpoints_to_lr_decrease)
    assert params.lr_checkpoint_increase_factor >= 1, "Parameter \'checkpoints_to_lr_decrease\' must be greater or equal to one. Received {}".format(params.lr_checkpoint_increase_factor)
    assert params.lr_decrease_factor > 0 and params.lr_decrease_factor <= 1, "Learning rate decrease factor must be in the interval 0 < param <= 1. Received {}".format(params.lr_decrease_factor)
    assert params.min_lr > 0 and params.min_lr <= 1, "Learning rate decrease factor must be in the interval 0 < param <= 1. Received {}".format(params.min_lr)
        
    # assert params.device_id >= 0, "Device ID must be greater than zero. Received {}".format(params.device_id)

    import os
    params.dataset_path = os.path.expandvars(params.dataset_path)
    params.save_path = os.path.expandvars(params.save_path)

    assert os.path.exists(params.dataset_path), "Dataset path '{}' does not exist".format(params.dataset_path)
    
    if not os.path.isabs(params.dataset_path):
        params.dataset_path = os.path.abspath('.' + os.path.sep + params.dataset_path)

    if not os.path.isabs(params.save_path):
        params.save_path = os.path.abspath('.' + os.path.sep + params.save_path)

    os.makedirs(params.save_path)
    os.makedirs(params.save_path + "/TrainedFlow")
    os.makedirs(params.save_path + "/TrainLosses")
    os.makedirs(params.save_path + "/TestLosses")
    os.makedirs(params.save_path + "/Params")
    os.makedirs(params.save_path + "/SampleImages")

    params.file_extension = '.' + params.file_extension


    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)


    sys.stderr.write('Available devices {}\n'.format(torch.cuda.device_count()))
    sys.stderr.write('Current cuda device {}\n'.format(torch.cuda.current_device()))

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:{}".format(params.device_id) if params.use_cuda and use_cuda else "cpu")
    device = torch.device("cuda:0" if params.use_cuda and use_cuda else "cpu")
    torch.multiprocessing.set_start_method('spawn')
    params.device = device.type.upper()

    # device_report = "Device: '{}'\n".format(params.device) + "{}".format(params.device_id) if params.use_cuda else ""
    device_report = "Device: '{}'\n".format(params.device) + "{}".format(0) if params.use_cuda else ""
    sys.stderr.write(device_report)

    return params, device

def load_data_generator(params, device):
    from torch.utils.data.dataloader import default_collate
    from tcc.data import CelebALoader

    using_cuda = device.type.upper() == "CUDA"
    loader_params = {
        'batch_size': params.batch_size,
        'shuffle': True,
        # 'collate_fn': lambda x: default_collate(x).to(device),
        'num_workers': params.num_threads,
        'pin_memory': False
        # 'persistent_workers': using_cuda
    }

    train_gen = None
    test_gen = None
    if params.dataset_loader == 'celeba':
        train_gen = CelebALoader(params, "train", **loader_params)
        test_gen = CelebALoader(params, "test", **loader_params)
    else:
        raise Exception("Invalid dataset loader parameter '{}'".format(params.dataset_loader))
    
    return train_gen, test_gen

def fill_left(string, col_w=15):
    left_space = (col_w - len(string))//2
    left_space = 0 if left_space < 0 else left_space
    string = " " * left_space + string
    return string

if __name__ == "__main__":
    params, device = parse()
    
    from tcc.model import Glow
    from tqdm import tqdm
    from torch import optim

    train_gen, test_gen = load_data_generator(params, device)
    
    if params.max_samples == 0:
        params.max_samples = len(train_gen)

    max_epochs = params.max_epochs

    img_shape = list(train_gen._dataset[0][0].shape)
    flow = Glow(img_shape[:2], num_scales=params.num_scales, num_flows=params.num_flows, conv_mid_channel_num=params.conv_mid_channel_num, pre_transform=Dequantization(), distribution=params.base_distribution).to(device)
    t_optimizer = optim.AdamW(flow._transform.parameters(), lr=params.learning_rate, weight_decay=0.1)
    
    d_optimizer = None
    if sum(param.numel() for param in flow._distribution.parameters()) > 0:
        d_optimizer = optim.Adam(flow._distribution.parameters(), lr=params.learning_rate)
    
    # optimizer = optim.SGD(flow.parameters(), lr=params.learning_rate)

    module_path = params.save_path + "/TrainedFlow/{}".format(params.session_name)
    train_losses_path = params.save_path + "/TrainLosses/{}".format(params.session_name)
    test_losses_path = params.save_path + "/TestLosses/{}".format(params.session_name)
    params_path = params.save_path + "/Params/{}".format(params.session_name)

    imgs_path = module_path.replace('/TrainedFlow', '/SampleImages')

    print(
            "{\n" \
            + "\t\"paths\": [\n\t\t  \"{}\",\n\t\t  \"{}\",\n\t\t  \"{}\",\n\t\t  \"{}\"\n\t\t]" \
            .format(
                module_path,
                train_losses_path,
                test_losses_path,
                params_path
            )
    )

    torch.save(flow, module_path + "+epoch0.pt")
    with open(params_path + ".json", 'w') as fp:
        json.dump(vars(params), fp)

    print("}\n\n", flush=True)

    print("Initiating training")

    checkpoint_frequency = 8192
    
    last_loss = torch.inf
    loss_not_dec_counter = 0
    total_samples = 0
    for epoch in range(max_epochs):
        train_losses = []
        test_losses = []
        print("\n =============== Epoch {}/{} ===============".format(epoch+1, max_epochs), flush=True)
        print( \
            "{:16} {:16} {:16} {:16}" \
            .format( \
                fill_left('Samples'), \
                fill_left('TrainLoss'), \
                fill_left('TestLoss'), \
                fill_left('LearningRate')
            ) \
        )

        iter_ = 0
        for x, idx in tqdm(train_gen):
            x = x.to(device).float()
            t_optimizer.zero_grad()
            if d_optimizer:
                d_optimizer.zero_grad()
            loss = -flow.log_prob(x).mean().to(device) / ( flow.latent_space_dims[0] )
            loss.backward()
            t_optimizer.step()
            if d_optimizer:
                d_optimizer.step() 

            train_losses.append(loss.cpu().item())

            if iter_ % checkpoint_frequency == 0 :
                last_n_samples = iter_
                next_n_samples = iter_ + x.shape[0]
                
                # Checks if should decrease learning rate
                if train_losses[-1] > last_loss:
                    loss_not_dec_counter += 1
                    
                    if loss_not_dec_counter == params.checkpoints_to_lr_decrease:
                        loss_not_dec_counter = 0
                        if params.min_lr <= t_optimizer.param_groups[0]['lr']:
                            last_loss = train_losses[-1]
                            t_optimizer.param_groups[0]['lr'] *= params.lr_decrease_factor
                            if d_optimizer:
                                d_optimizer.param_groups[0]['lr'] *= params.lr_decrease_factor
                            params.checkpoints_to_lr_decrease = int(params.checkpoints_to_lr_decrease * params.lr_checkpoint_increase_factor)

                            if params.min_lr <= t_optimizer.param_groups[0]['lr']:
                                t_optimizer.param_groups[0]['lr'] = params.min_lr
                                if d_optimizer:
                                    d_optimizer.param_groups[0]['lr'] = params.min_lr
                        else:
                            print("Warning: tried to reduce learning rate, but it has reached its minimum value.")

                else:
                    last_loss = loss.item()
                    loss_not_dec_counter = 0
            
                # Outputs training info
                with torch.no_grad():
                    test_batch, indices = next(iter(test_gen))
                    test_loss = -flow.log_prob(test_batch.to(device).float()).mean() / ( flow.latent_space_dims[0] )
                    test_losses.append(test_loss.cpu().item())

                    if (last_n_samples % (2*checkpoint_frequency)) == 0:
                        img = build_horizontal_image_sequence([transforms.ToPILImage()(s.to(torch.uint8)) for s in flow.sample_tempered_dist(10, 0.8)[0]])
                        img.save("{}+epoch{}+s{}.png".format(imgs_path, epoch, next_n_samples), "PNG")

                print( \
                    "{:16} {:16} {:16} {:16}" \
                    .format( \
                        fill_left("{:d}".format(total_samples)), \
                        fill_left("{:f}".format(train_losses[-1])), \
                        fill_left("{:f}".format(test_losses[-1])), \
                        fill_left("{:f}".format(t_optimizer.param_groups[0]['lr']))
                    ), \
                    flush=True \
                )

                total_samples += checkpoint_frequency


            iter_ += x.shape[0]

            # TODO this is possibly causing problems with respect to the data loader.
            # The data loader should be the sole responsible for this loop
            if iter_ >= params.max_samples:
                break


        torch.save(flow, module_path + "+epoch{}.pt".format(epoch+1))
        torch.save(train_losses, train_losses_path + "+epoch{}.pt".format(epoch+1))
        torch.save(test_losses, test_losses_path + "+epoch{}.pt".format(epoch+1))
    
    print("\nTraining Finished")

