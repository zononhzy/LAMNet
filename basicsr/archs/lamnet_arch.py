import torch
import torch.nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.arch_util import trunc_normal_

from fsa import fsa_spatial


class LayerNormUnchanged(nn.Module):
    """
    LayerNormUnchanged applies layer normalization to the input tensor without changing its shape.

    Args:
        dim (int): The number of input channels.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-05.

    Attributes:
        eps (float): A value added to the denominator for numerical stability.
        alpha (nn.Parameter): A learnable scale parameter.
        beta (nn.Parameter): A learnable shift parameter.
    """

    def __init__(self, dim, eps=1e-05):
        super(LayerNormUnchanged, self).__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, correction=0).add(self.eps).sqrt()
        return self.alpha * (x - mean) / std + self.beta


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormUnchanged(dim)
        self.fn = fn
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return self.fn(self.norm(x)) * self.alpha + x


class PostNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormUnchanged(dim)
        self.fn = fn
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return self.norm(self.fn(x)) * self.alpha + x


class DGFN(nn.Module):
    """
    DGFN (Dual-Gate Feedforward Network) in Restormer: [github] https://github.com/swz30/Restormer

    Args:
        dim (int): The number of input and output channels.
        expansion_factor (float): The factor by which the hidden dimension is expanded.
        bias (bool): If True, adds a learnable bias to the output.

    Attributes:
        dim (int): The number of input and output channels.
        expansion_factor (float): The factor by which the hidden dimension is expanded.
        project_in (nn.Conv2d): Convolutional layer to project input to hidden dimension.
        dwconv (nn.Conv2d): Depthwise convolutional layer.
        project_out (nn.Conv2d): Convolutional layer to project hidden dimension back to input dimension.
    """

    def __init__(self, dim, expansion_factor, bias):
        super(DGFN, self).__init__()

        self.dim = dim
        self.expansion_factor = expansion_factor

        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        gate_x1 = F.gelu(x1) * x1
        gate_x2 = F.gelu(x1) * x2
        return self.project_out(torch.cat([gate_x1, gate_x2], dim=1))


class SpatialInteraction(nn.Module):
    """
    Information Exchange Module applies spatial interaction between channel and spatial maps.

    Attributes:
        activation (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self):
        super(SpatialInteraction, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, channel_map, spatial_map):
        """
        Args:
            channel_map (Tensor): (B, C, HW)
            spatial_map (Tensor): (B, C, HW)

        Returns:
            Tensor: (B, C, HW)
        """
        spatial_map = spatial_map.mean(dim=-1).unsqueeze(1)
        attn = self.activation(spatial_map @ channel_map / spatial_map.size(2))
        return attn * channel_map


class ChannelInteraction(nn.Module):
    """
    Information Exchange Module applies channel interaction between channel and spatial maps.

    Attributes:
        activation (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self):
        super(ChannelInteraction, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, channel_map, spatial_map):
        """
        Args:
            channel_map (Tensor): (B, C, HW)
            spatial_map (Tensor): (B, C, HW)

        Returns:
            Tensor: (B, C, HW)
        """
        channel_map = channel_map.mean(dim=1).unsqueeze(2)
        attn = self.activation(spatial_map @ channel_map / spatial_map.size(2))
        return attn * spatial_map


class LSAMixer(nn.Module):
    """
    LSAMixer (Local Spatial Attention Mixer) applies local spatial attention mixing to the input tensor.

    Args:
        dim (int): The number of input channels.
        kernel_size (int): The size of the convolutional kernel.
        kernel_loc (int): The start locations of the kernel.
        kernel_stride (int): The strides of the convolutional kernel.
        num_heads (int): The number of heads for the local attention mechanism.
    """

    def __init__(
        self,
        dim,
        kernel_size,
        kernel_loc,
        kernel_stride,
        num_heads,
    ):
        super(LSAMixer, self).__init__()
        self.kernel_size = kernel_size
        self.groups = num_heads
        self.dim = dim // 2
        self.groups_dim = dim // self.groups // 2

        kernel_map, patch_size = self.generate_kernel_map_1d(
            kernel_size, kernel_stride, kernel_loc
        )
        self.register_buffer("kernel_map", kernel_map)
        self.patch_size = patch_size

        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.h_kernel = nn.Sequential(
            nn.Conv2d(
                dim // 2,
                dim // 2,
                (1, self.patch_size),
                padding=(0, self.patch_size // 2),
                groups=dim // 2,
            ),
            nn.Conv2d(dim // 2, self.groups * kernel_size, 1),
        )

        self.v_kernel = nn.Sequential(
            nn.Conv2d(
                dim // 2,
                dim // 2,
                (self.patch_size, 1),
                padding=(self.patch_size // 2, 0),
                groups=dim // 2,
            ),
            nn.Conv2d(dim // 2, self.groups * kernel_size, 1),
        )
        self.channel_branch = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 1),
        )
        self.spatial_interaction = SpatialInteraction()
        self.channel_interaction = ChannelInteraction()

        self.proj_out = nn.Conv2d(dim, dim, 1)

    def generate_kernel_map_1d(self, kernel_size, kernel_stride, kernel_loc):
        """
        Args:
            kernel_size (list): kernel size kernel
            kernel_stride (list): kernel stride for different locations, [1, 2, 4,...]
            kernel_loc (list): stride change location for one side, len(kernel_loc) = len(kernel_stride) [1, 3, 5,...],
        """
        assert len(kernel_stride) == len(kernel_loc)
        assert kernel_loc[-1] == kernel_size // 2 + 1

        def get_grid_1d(kernel_loc, kernel_stride):
            grid_axes_side = [0]  # for center
            grid_stride_side = [1]  # for center
            kernel_start = 1
            for loc, stride in zip(kernel_loc, kernel_stride):
                grid_axes_side.extend(
                    [
                        num * stride + grid_axes_side[-1] + grid_stride_side[-1]
                        for num in range(0, loc - kernel_start)
                    ]
                )
                grid_stride_side.extend([stride] * (loc - kernel_start))
                kernel_start = loc
            patch_size = (grid_axes_side[-1] + grid_stride_side[-1]) * 2 - 1
            grid_axes = [-num for num in grid_axes_side[:0:-1]] + grid_axes_side
            grid_stride_axes = grid_stride_side[:0:-1] + grid_stride_side
            grid_axes = torch.tensor(grid_axes, dtype=torch.int)
            grid_stride_axes = torch.tensor(grid_stride_axes, dtype=torch.int)

            return grid_axes, grid_stride_axes, patch_size

        grid_axes, grid_stride_axes, patch_size = get_grid_1d(kernel_loc, kernel_stride)

        kernel_map = torch.stack([grid_axes, grid_stride_axes], dim=1)

        return kernel_map, patch_size

    def forward(self, x):
        B, C, H, W = x.size()
        x1, x2 = self.proj_in(x).chunk(2, dim=1)
        x_conv = self.channel_branch(x1)
        kernel_h = F.tanh(self.h_kernel(x2))
        attn_h = fsa_spatial(
            x2.contiguous(),
            kernel_h.contiguous(),
            self.kernel_size,
            self.kernel_map,
            self.groups,
            self.groups_dim,
            direction="horizontal",
        )

        kernel_v = F.tanh(self.v_kernel(attn_h))
        attn = fsa_spatial(
            attn_h.contiguous(),
            kernel_v.contiguous(),
            self.kernel_size,
            self.kernel_map,
            self.groups,
            self.groups_dim,
            direction="vertical",
        )
        x_conv = x_conv.view(B, self.dim, H * W)
        attn = attn.view(B, self.dim, H * W)
        spatial_map = self.spatial_interaction(x_conv, attn).view(B, self.dim, H, W)
        channel_map = self.channel_interaction(x_conv, attn).view(B, self.dim, H, W)
        return self.proj_out(torch.cat([spatial_map, channel_map], dim=1))


class LAMixerBlock(nn.Module):
    """
    LAMixerBlock applies a series of  Linear-Spatial Adaptive Mixer (LSAMixer) and Dual-Gate Feedforward Networks (DGFN) 
    to the input tensor.

    Args:
        dim (int): The number of input channels.
        num_blocks (int): The number of mixer blocks.
        kernel_size (int): The size of the convolutional kernel.
        kernel_loc (list, optional): The start locations of the kernel. Default: [3, 4, 5].
        kernel_stride (list, optional): The strides of the convolutional kernel. Default: [1, 2, 4].
        num_head (int, optional): The number of heads for the local attention mechanism. Default: 4.
        expansion_factor (float, optional): The factor by which the hidden dimension is expanded. Default: 1.0.

    Attributes:
        mixers (nn.Sequential): A sequential container of PreNormResidual blocks containing LSAMixer and DGFN.
        alpha (nn.Parameter): A learnable parameter for scaling the output.
    """
    def __init__(
        self,
        dim,
        num_blocks,
        kernel_size,
        kernel_loc=[3, 4, 5],
        kernel_stride=[1, 2, 4],
        num_head=4,
        expansion_factor=1,
    ):
        super(LAMixerBlock, self).__init__()

        body = []
        for _ in range(num_blocks):
            body.append(
                PreNormResidual(
                    dim,
                    LSAMixer(
                        dim,
                        kernel_size,
                        kernel_loc,
                        kernel_stride,
                        num_head,
                    ),
                )
            )
            body.append(
                PreNormResidual(
                    dim,
                    DGFN(
                        dim,
                        expansion_factor,
                        bias=True,
                    ),
                )
            )
        self.mixers = nn.Sequential(*body)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return self.mixers(x) * self.alpha + x


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

       but for our model, we give up Traditional Upsample and use UpsampleOneStep for better performance not only in
       lightweight SR model, Small/XSmall SR model, but also for our base model.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


@ARCH_REGISTRY.register()
class LAMNet(nn.Module):
    """
    LAMNet (Linear Adaptive Mixer Network) is a neural network architecture designed for image super-resolution tasks.

    Args:
        in_chans (int): Number of input channels.
        num_blocks (int): Number of mixer blocks in each group.
        num_groups (int): Number of groups of mixer blocks.
        dim (int): Number of feature dimensions.
        kernel_size (int): Size of the convolutional kernel.
        kernel_loc (list): Locations of the kernel.
        kernel_stride (list): Strides of the convolutional kernel.
        num_head (int): Number of heads for the local attention mechanism.
        expansion_factor (float): Factor by which the hidden dimension is expanded.
        upscale (int): Upscaling factor for the output image.
        img_range (float): Range of the image values.
        rgb_mean (tuple): Mean RGB values for normalization.
    """
    def __init__(
        self,
        in_chans=3,
        num_blocks=4,
        num_groups=4,
        dim=48,
        kernel_size=11,
        kernel_loc=[3, 4, 5],
        kernel_stride=[1, 2, 4],
        num_head=4,
        expansion_factor=1.0,
        upscale=4,
        img_range=1.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        **kwargs
    ):
        super(LAMNet, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.shallow_feature_extraction = nn.Conv2d(in_chans, dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        m_body = []

        for _ in range(num_groups):
            m_body.append(
                LAMixerBlock(
                    dim,
                    num_blocks,
                    kernel_size,
                    kernel_loc,
                    kernel_stride,
                    num_head,
                    expansion_factor,
                )
            )
        m_body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.deep_feature_extraction = nn.Sequential(*m_body)

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

        self.upsample = UpsampleOneStep(upscale, dim, in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.shallow_feature_extraction(x)
        x = self.deep_feature_extraction(x) * self.alpha + x

        x = self.upsample(x)

        x = x / self.img_range + self.mean

        return x

if __name__ == "__main__":
    upscale = 4
    height = 1280 // upscale
    width = 720 // upscale
    input = torch.randn(1, 3, height, width).cuda()

    model = LAMNet(
        in_chans=3,
        num_blocks=6,
        num_groups=4,
        dim=64,
        kernel_size=13,
        kernel_loc=[4, 6, 7],
        kernel_stride=[1, 2, 4],
        num_head=4,
        expansion_factor=1.0,
        upscale=upscale,
        img_range=1.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ).cuda()
    model.eval()

    with torch.no_grad():

        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

        flops_count = FlopCountAnalysis(model, (input,))
        print(flop_count_str(flops_count))

        import numpy as np
        import tqdm

        repetitions = 10

        print("warm up ...\n")
        with torch.no_grad():
            for _ in range(10):
                _ = model(input)

        torch.cuda.synchronize()

        # testing CUDA Event
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        # initialize
        timings = np.zeros((repetitions, 1))

        print("testing ...\n")
        with torch.no_grad():
            for rep in tqdm.tqdm(range(repetitions)):
                starter.record()
                _ = model(input)
                ender.record()
                torch.cuda.synchronize()  # wait for ending
                curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
                timings[rep] = curr_time

        avg = timings.sum() / repetitions
        print("\navg={}\n".format(avg))

        print(flop_count_table(flops_count))
