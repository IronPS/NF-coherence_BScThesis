
import torch
import math

class NormalGamma:
    def __init__(self, m, l, a, b, device=None):
        self._m0 = torch.tensor(m, device=device)
        self._l0 = torch.tensor(l, device=device)
        self._a0 = torch.tensor(a, device=device)
        self._b0 = torch.tensor(b, device=device)

        self._m = self._m0
        self._l = self._l0
        self._a = self._a0
        self._b = self._b0

        self._fb_predictive_mean = None

    def pdf(self, m, l):
        return torch.exp(self.lpdf(m, l))

    def lpdf(self, m, l):
        device = self._a.device

        m = m.to(device)
        l = l.to(device)
        res = self._a * torch.log(self._b) + 0.5 * torch.log(self._l)
        res = res - torch.lgamma(self._a) + 0.5 * torch.log(torch.tensor(2*math.pi, device=device))

        res = res + (self._a - 0.5) * torch.log(l)
        res = res - self._b * l
        res = res - 0.5 * self._l * l * torch.square(m - self._m)

        return res

    def fit(self, X):
        """
        Calculates a normal-gamma distribution for each dimension of X independently
        Parameters:
            X: A tensor (N, M) of N samples and M dimensions
        """
        device = self._a.device

        X = X.to(device)
        N = X.shape[0]
        s2_mle, m_mle = torch.var_mean(X, dim=0, unbiased=False)

        self._m = self._l0 + N * m_mle
        self._m /= self._l0 + N

        self._l = self._l0 + N
        
        self._a = self._a0 + N * 0.5
        
        self._b = self._b0 + 0.5 * (N * s2_mle + self._l0 * N * torch.square(m_mle - self._m0) / ( self._l0 + N ) )

        return self
    
    def mean_mean(self):
        return self._m

    def mode_precision(self):
        return (self._a - 1) / self._b


        