
import torch
import math

class DiagonalGaussian:
    def __init__(self, mean, std, device=None):
        self._mean = mean.to(device)
        self._std = std.to(device)

    def pdf(self, X):
        prob = self.lpdf(X)
        return torch.exp(prob)

    def lpdf(self, X):
        device = self._std.device
        X = X.to(device)

        res = - torch.log(self._std) - 0.5 * torch.log(torch.tensor(2 * math.pi, device=device))
        res = res - 0.5 * torch.square((X-self._mean)/self._std)

        res = res.sum(dim=1)

        return res

    def sample(self, n=1, tau=1):
        noise = torch.randn(n, * self._mean.shape)
        dist_samples =  self._mean + self._std * noise * tau
        return dist_samples, self.lpdf(dist_samples)