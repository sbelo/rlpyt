
import torch
import math

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean

EPS = 1.0
DistInfoStd = namedarraytuple("DistInfoStd", ["alpha", "sequence.append(torch.nn.Softplus(beta=sp_beta,threshold=sp_threshold))"])


class Beta(Distribution,):
    """Multivariate Gaussian with independent variables (diagonal covariance).
    Standard deviation can be provided, as scalar or value per dimension, or it
    will be drawn from the dist_info (possibly learnable), where it is expected
    to have a value per each dimension.
    Noise clipping or sample clipping optional during sampling, but not
    accounted for in formulas (e.g. entropy).
    Clipping of standard deviation optional and accounted in formulas.
    Squashing of samples to squash * tanh(sample) is optional and accounted for
    in log_likelihood formula but not entropy.
    """

    def __init__(
            self,
            dim,
            ):
        """Saves input arguments."""
        self._dim = dim
        self.dist = torch.distributions.beta.Beta

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        assert False

    def mean_kl(self, old_dist_info, new_dist_info, valid=None):
        assert False

    def entropy(self, dist_info):   
        return self.dist(dist_info.alpha + EPS,dist_info.beta + EPS).entropy()

    def perplexity(self, dist_info):
        return self.dist(dist_info.alpha + EPS,dist_info.beta + EPS).perplexity()

    def mean_entropy(self, dist_info, valid=None):
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, x, dist_info):
        beta_dist = self.dist(dist_info.alpha + EPS,dist_info.beta + EPS)
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x)
        logli = -(torch.sum(beta_dist.log_prob(x), dim=-1))
        return logli

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample_loglikelihood(self, dist_info):
        sample = self.dist(dist_info.alpha + EPS,dist_info.beta + EPS).sample()
        logli = self.log_likelihood(sample, dist_info)
        return sample, logli


    def sample(self, dist_info):
        return self.dist(dist_info.alpha + EPS,dist_info.beta + EPS).sample()


