import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .weight_noise import noise_fn

# Set device to use CUDA if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandConv2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(RandConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).to(device))
        self.weight = None
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels).to(device))
            self.biasp = None
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_parameter('eps_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def forward_(self, input):
        self.weight = noise_fn(self.mu_weight, self.sigma_weight, self.eps_weight, self.sigma_0, self.N)
        self.biasp = None
        if self.mu_bias is not None:
            self.biasp = noise_fn(self.mu_bias, self.sigma_bias, self.eps_bias, self.sigma_0, self.N)
        out = F.conv2d(input, self.weight, self.biasp, self.stride, self.padding, self.dilation, self.groups)
        return out

    def forward(self, input, sample=True, fix=False):
        if not sample:
            out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride, self.padding, self.dilation, self.groups)
            return out, 0
        sig_weight = torch.exp(self.sigma_weight)

        if fix:
            eps_weight = self.eps_weight
        else:
            eps_weight = torch.Tensor(*self.eps_weight.size()).normal_().to(device)
            assert eps_weight.shape == self.eps_weight.shape

        self.weight = self.mu_weight + sig_weight * eps_weight

        kl_weight = math.log(self.sigma_0) - self.sigma_weight + (sig_weight ** 2 + self.mu_weight ** 2) / (
                    2 * self.sigma_0 ** 2) - 0.5
        self.biasp = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)

            if fix:
                eps_bias = self.eps_bias
            else:
                eps_bias = torch.Tensor(*self.eps_bias.size()).normal_().to(device)
                assert eps_bias.shape == self.eps_bias.shape

            self.biasp = self.mu_bias + sig_bias * eps_bias

            kl_bias = math.log(self.sigma_0) - self.sigma_bias + (sig_bias ** 2 + self.mu_bias ** 2) / (
                        2 * self.sigma_0 ** 2) - 0.5
        out = F.conv2d(input, self.weight, self.biasp, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
        return out, kl
