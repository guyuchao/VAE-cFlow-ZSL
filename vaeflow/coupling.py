import torch.nn as nn
from vaeflow.actnorm import ActNorm2d
from vaeflow.inveitable import InvertibleConv1x1
import torch
import vaeflow.thops as thops
import numpy as np
class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[1, 1], stride=[1, 1],logscale_factor=3):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=[1, 1], stride=[1, 1],do_actnorm=True, weight_std=0.05):
        super().__init__(in_channels, out_channels, kernel_size, stride,bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x

def f(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        Conv2d(in_channels, hidden_channels,kernel_size=[1, 1]), nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]), nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels,kernel_size=[1, 1]))


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1).long()
        self.indices_inverse = torch.zeros((self.num_channels)).long()
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()
        self.register_buffer('tindices',self.indices)
        self.register_buffer('tindices_inverse',self.indices_inverse )


    def reset_indices(self):
        self.indices=self.indices[torch.randperm(self.num_channels)]
        #np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4
        if not reverse:
            return input[:, self.tindices, :, :]
        else:
            return input[:, self.tindices_inverse, :, :]

'''
class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4
        if not reverse:
            return input[:, self.indices, :, :]
        else:
            return input[:, self.indices_inverse, :, :]
'''

class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="affine",
                 LU_decomposed=True):
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet

        z = thops.cat_feature(z1, z2)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet