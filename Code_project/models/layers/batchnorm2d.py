import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .weight_noise import noise_fn

# Dynamically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandBatchNorm2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(RandBatchNorm2d, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.num_features = num_features
        self.init_s = init_s
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.mu_weight = Parameter(torch.Tensor(num_features))
            self.sigma_weight = Parameter(torch.Tensor(num_features))
            self.register_buffer('eps_weight', torch.Tensor(num_features).to(device))
            self.mu_bias = Parameter(torch.Tensor(num_features))
            self.sigma_bias = Parameter(torch.Tensor(num_features))
            self.register_buffer('eps_bias', torch.Tensor(num_features).to(device))
            self.weight = None
            self.biasp = None
        else:
            self.register_parameter('mu_weight', None)
            self.register_parameter('sigma_weight', None)
            self.register_buffer('eps_weight', None)
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_buffer('eps_bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features).to(device))
            self.register_buffer('running_var', torch.ones(num_features).to(device))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(device))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.mu_weight.data.uniform_()
            self.sigma_weight.data.fill_(self.init_s)
            self.mu_bias.data.zero_()
            self.sigma_bias.data.fill_(self.init_s)
            self.eps_weight.data.zero_()
            self.eps_bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward_(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        # generate weight and bias
        self.weight = self.biasp = None
        if self.affine:
            self.weight = noise_fn(self.mu_weight, self.sigma_weight, self.eps_weight, self.sigma_0, self.N)
            self.biasp = noise_fn(self.mu_bias, self.sigma_bias, self.eps_bias, self.sigma_0, self.N)

        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.biasp,
                            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def forward(self, input, sample=True, fix=False):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        # generate weight and bias
        if not sample:
            out = F.batch_norm(input, self.running_mean, self.running_var, self.mu_weight, self.mu_bias,
                               self.training or not self.track_running_stats, exponential_average_factor, self.eps)
            return out, 0
        self.weight = self.biasp = None
        if self.affine:
            sig_weight = torch.exp(self.sigma_weight)

            if fix:
                eps_weight = self.eps_weight
            else:
                eps_weight = torch.Tensor(*self.eps_weight.size()).normal_().to(device)
                assert eps_weight.shape == self.eps_weight.shape

            self.weight = self.mu_weight + sig_weight * eps_weight

            # KL Divergence for weight
            kl_weight = (math.log(self.sigma_0) - self.sigma_weight + 
                         (sig_weight ** 2 + self.mu_weight ** 2) / (2 * self.sigma_0 ** 2) - 0.5)
            
            sig_bias = torch.exp(self.sigma_bias)

            if fix:
                eps_bias = self.eps_bias
            else:
                eps_bias = torch.Tensor(*self.eps_bias.size()).normal_().to(device)
                assert eps_bias.shape == self.eps_bias.shape

            self.biasp = self.mu_bias + sig_bias * eps_bias

            # KL Divergence for bias
            kl_bias = (math.log(self.sigma_0) - self.sigma_bias + 
                       (sig_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma_0 ** 2) - 0.5)

        out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.biasp,
                           self.training or not self.track_running_stats, exponential_average_factor, self.eps)
        kl = kl_weight.sum() + kl_bias.sum()
        return out, kl
