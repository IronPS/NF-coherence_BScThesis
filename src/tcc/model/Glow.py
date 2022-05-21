from setuptools import Distribution
import torch
from torch import nn 
from torch.nn import init

from nflows.flows.base import Flow

from nflows.transforms.base import CompositeTransform, MultiscaleCompositeTransform
from nflows.transforms.reshape import SqueezeTransform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.conv import OneByOneConvolution
from nflows.transforms.coupling import AffineCouplingTransform

from nflows.distributions.normal import DiagonalNormal, StandardNormal

class ThreeConvNet(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 3, padding=1), # 3x3: in -> mid_channel
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 1), # 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, 3, padding=1), # 3x3: mid_channel -> out
        )
        
        init.uniform_(self.net[0].weight, -1e-3, 1e-3)
        init.uniform_(self.net[0].bias, -1e-3, 1e-3)
        init.uniform_(self.net[2].weight, -1e-3, 1e-3)
        init.uniform_(self.net[2].bias, -1e-3, 1e-3)
        init.constant_(self.net[4].weight, 0)
        init.constant_(self.net[4].bias, 0)
        
    def forward(self, input, context=None):
        return self.net(input)

class Glow(Flow):
    def __init__(
        self,
        dims,
        num_scales=3,
        num_flows=32,
        conv_mid_channel_num=512,
        pre_transform=None,
        distribution='std_normal'
    ):
        """

        Parameters
        ----------
            dims: image dimensions in (Channels, Size)
                Here we consider only the case where image width and height are equal
            num_flows: the number of flows contained in a scale
            num_scales: the number of scales
            conv_mid_channel_num: the number of channels of the central layer of the scale network
            distribution: 'std_normal', 'diagonal_normal'
        """

        from functools import reduce as Reduce
        self.latent_space_dims = [Reduce(lambda x, y: x*y, dims)*dims[-1]]
        self.img_channels, self.img_size = dims

        # init net flow
        crop_size_denom = 2
        num_channel_factor = int(2 * crop_size_denom)
        output_channels = int(self.img_channels * num_channel_factor)
        output_crop_size = int(self.img_size // crop_size_denom)
        
        def create_net(in_channels, out_channels, mid_channel=conv_mid_channel_num):
            return ThreeConvNet(in_channels, out_channels, mid_channel)
        
        transform = MultiscaleCompositeTransform(num_scales)
        for i in range(num_scales):
            assert output_crop_size >= 4, "Image too small for split no. '{}': found {}. Try providing bigger images or reducing the number of scales.".format(i+1, curr_crop_size)

            mask = torch.ones((output_channels))
            mask[::2] = -1
        
            transforms = [SqueezeTransform()]
            for _ in range(num_flows):
                transforms += [
                    ActNorm(output_channels),
                    OneByOneConvolution(output_channels),
                    AffineCouplingTransform(mask=mask, transform_net_create_fn=create_net)
                ]

                mask *= -1
            
            transform.add_transform(CompositeTransform(transforms), [output_channels, output_crop_size, output_crop_size])
            
            
            output_channels *= int(num_channel_factor // 2) # Account for reduced data
            output_crop_size //= int(crop_size_denom)
            
            
        if pre_transform:
            transform = CompositeTransform([pre_transform, transform])

        dist = None
        if not distribution or distribution == 'std_normal':
            dist = StandardNormal(shape=self.latent_space_dims)
        elif distribution == 'diagonal_normal':
            dist = DiagonalNormal(shape=self.latent_space_dims)

        super().__init__(
            transform=transform,
            distribution=dist
        )

    def sample_tempered_dist(self, n=1, tau=0.7):
        samples, ldjs = None, None
        
        lats, _ = self.get_tempered_latent_samples(n=n, tau=tau)

        samples, ldjs = self._transform.inverse(lats)

        return samples, ldjs

    def get_tempered_latent_samples(self, n=1, tau=0.7):
        if type(self._distribution) == DiagonalNormal:
            noise = torch.randn(n, * self._distribution._shape, device=self._distribution.mean_.device)
            dist_samples = self._distribution.mean_ + torch.exp(self._distribution.log_std_) * noise * tau
            return dist_samples, self._distribution._log_prob(dist_samples, None)

        elif type(self._distribution) == StandardNormal:
            dist_samples = tau * self._distribution.sample(num_samples=n)
            return dist_samples, self._distribution._log_prob(dist_samples, None)

        else:
            raise NotImplementedError()

    def mean_(self):
        if type(self._distribution) == DiagonalNormal:
            return self._distribution.mean_.reshape(1, -1)
        
        elif type(self._distribution) == StandardNormal:
            return self._distribution._mean(None).reshape(1, -1)

        else:
            raise NotImplementedError()

    def std_(self):
        if type(self._distribution) == DiagonalNormal:
            return torch.exp(self._distribution.log_std_).reshape(1, -1)
        
        elif type(self._distribution) == StandardNormal:
            return torch.ones_like(self._distribution._mean(None)).reshape(1, -1)

        else:
            raise NotImplementedError()