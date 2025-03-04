import torch

import torch.nn as nn
import torch.nn.functional as F


class SparseConvHead(nn.Module):
    def __init__(self, in_channels, out_channels, sparsity=0.5):
        super(SparseConvHead, self).__init__()
        self.sparsity = sparsity
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.mask = self._create_mask(out_channels)

    def _create_mask(self, out_channels):
        mask = torch.ones(out_channels, out_channels, 3, 3)
        num_elements = mask.numel()
        num_zeros = int(num_elements * self.sparsity)
        indices = torch.randperm(num_elements)[:num_zeros]
        mask.view(-1)[indices] = 0
        return mask

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        self.conv2.weight.data *= self.mask.to(self.conv2.weight.device)
        x = self.conv2(x)
        return x


class DepthwiseSeparableConvHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvHead, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        return x

# Example usage:
# sparse_conv_head = SparseConvHead(in_channels=3, out_channels=64, sparsity=0.5)
# depthwise_separable_conv_head = DepthwiseSeparableConvHead(in_channels=3, out_channels=64)
