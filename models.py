import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class NeRVBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, upscale_factor=1,
                 bias=True, activation='gelu'):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_chan, out_chan*(upscale_factor**2),
                              kernel_size, 1, kernel_size//2, bias=bias)
        self.up_scale = nn.PixelShuffle(upscale_factor)
        self.act = get_activation_fn(activation)

    def forward(self, x):
        outputs = self.act(self.up_scale(self.conv(x)))
        return outputs


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, upscale_factor=1,
                 bias=True, activation='gelu'):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv0 = nn.Conv2d(in_chan, out_chan*(upscale_factor**2),
                               1, 1, 0, bias=bias)
        self.up_scale = nn.PixelShuffle(upscale_factor)
        self.act = get_activation_fn(activation)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size, 1,
                               kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size, 1,
                               kernel_size//2, bias=bias)

    def forward(self, x):
        outputs = self.up_scale(self.conv0(x))
        outputs = self.act(outputs)
        outputs = self.conv1(outputs) * torch.tanh(self.conv2(outputs))
        return outputs


class MLP(nn.Module):
    def __init__(self, dim_list, activation='gelu', bias=True):
        super().__init__()

        self.mlp = []
        for i in range(len(dim_list) - 1):
            self.mlp += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias),
                         get_activation_fn(activation)]

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class Voxel(nn.Module):
    def __init__(self, resolution, channels, mode='bilinear', std=1e-4,
                 learn_pose=True, pose_mode='full'):
        """ resolution: a list of three elements ([time, height, width]) """
        # grid should be [W, H, T] (because of F.grid_sample)
        super().__init__()
        self.voxel = nn.Parameter(torch.randn(channels, *resolution) * std)
        self.t_map = nn.Parameter(
            torch.pi * (torch.arange(resolution[0]) + 0.5), requires_grad=False)
        self.h_map = nn.Parameter(
            torch.pi * (torch.arange(resolution[1]) + 0.5), requires_grad=False)
        self.w_map = nn.Parameter(
            torch.pi * (torch.arange(resolution[2]) + 0.5), requires_grad=False)
        self.resolution = resolution
        self.mode = mode
        self.pose_mode = pose_mode

        if pose_mode == 'full':
            self.pose_r = nn.Parameter(torch.randn(3, 3),
                                       requires_grad=learn_pose)
        elif pose_mode == 'partial':
            self.pose_r = nn.Parameter(torch.randn(2, 3),
                                       requires_grad=learn_pose)
        elif pose_mode == 'rodriguez':
            self.pose_r = nn.Parameter(torch.randn(3), requires_grad=learn_pose)
        else:
            raise ValueError(f'invalid pose_mode {pose_mode}')
        self.pose_t = nn.Parameter(torch.zeros((1, 1, 1, 3), dtype=torch.float),
                                   requires_grad=learn_pose)

    def translate(self, grid):
        # grid: [..., 3] shaped tensor
        if self.pose_mode == 'full':
            matrix = F.normalize(self.pose_r, dim=1)
        elif self.pose_mode == 'partial':
            matrix = F.normalize(self.pose_r, dim=1)
            matrix = torch.cat(
                [matrix, torch.cross(matrix[0], matrix[1])[None, :]], 0)
        else:
            matrix = torch.zeros((3, 3), dtype=torch.float, device=grid.device)
            matrix[0, 1] = -self.pose_r[2]
            matrix[0, 2] = self.pose_r[1]
            matrix[1, 0] = self.pose_r[2]
            matrix[1, 2] = -self.pose_r[0]
            matrix[2, 0] = -self.pose_r[1]
            matrix[2, 1] = self.pose_r[0]

            norm = matrix.norm().clamp(min=1e-15)

            matrix = torch.eye(3).to(grid.device) \
                   + torch.sin(self.pose_r[-1]) / norm * matrix \
                   + (1-torch.cos(self.pose_r[-1]))/(norm**2) * (matrix@matrix)
            matrix = matrix.transpose(1, 0)

        return grid @ matrix + self.pose_t

    def forward(self, grid):
        # assume every value in the grid is between -1 and 1
        grid = self.translate(grid)

        grid = (grid +1) / 2 # [-1, 1] to [0, 1]

        # voxel
        # grid: [batch, 3]
        outputs = torch.einsum('cthw,...w->...cth', self.voxel,
                               torch.cos(grid[..., 2:3] * self.w_map))
        outputs = torch.einsum('...cth,...h->...ct', outputs,
                               torch.cos(grid[..., 1:2] * self.h_map))
        outputs = torch.einsum('...ct,...t->...c', outputs,
                               torch.cos(grid[..., 0:1] * self.t_map))

        return outputs.permute(0, 3, 1, 2) # [B,H,W,C] -> [B,C,H,W]


class OldVoxel(nn.Module):
    def __init__(self, resolution, channels, mode='bilinear', std=1e-4,
                 learn_pose=True, pose_mode='full'):
        """ resolution: a list of three elements ([time, height, width]) """
        # grid should be [W, H, T] (because of F.grid_sample)
        super().__init__()
        self.voxel = nn.Parameter(
            torch.randn(1, channels, *reversed(resolution)) * std)
        self.mode = mode
        self.pose_mode = pose_mode

        if pose_mode == 'full':
            self.pose_r = nn.Parameter(torch.randn(3, 3),
                                       requires_grad=learn_pose)
        elif pose_mode == 'partial':
            self.pose_r = nn.Parameter(torch.randn(2, 3),
                                       requires_grad=learn_pose)
        elif pose_mode == 'rodriguez':
            self.pose_r = nn.Parameter(torch.randn(3), requires_grad=learn_pose)
        else:
            raise ValueError(f'invalid pose_mode {pose_mode}')
        self.pose_t = nn.Parameter(torch.zeros((1, 1, 1, 3), dtype=torch.float),
                                   requires_grad=learn_pose)

    def translate(self, grid):
        # grid: [..., 3] shaped tensor
        if self.pose_mode == 'full':
            matrix = F.normalize(self.pose_r, dim=1)
        elif self.pose_mode == 'partial':
            matrix = F.normalize(self.pose_r, dim=1)
            matrix = torch.cat(
                [matrix, torch.cross(matrix[0], matrix[1])[None, :]], 0)
        else:
            matrix = torch.zeros((3, 3), dtype=torch.float, device=grid.device)
            matrix[0, 1] = -self.pose_r[2]
            matrix[0, 2] = self.pose_r[1]
            matrix[1, 0] = self.pose_r[2]
            matrix[1, 2] = -self.pose_r[0]
            matrix[2, 0] = -self.pose_r[1]
            matrix[2, 1] = self.pose_r[0]

            norm = matrix.norm().clamp(min=1e-15)

            matrix = torch.eye(3).to(grid.device) \
                   + torch.sin(self.pose_r[-1]) / norm * matrix \
                   + (1-torch.cos(self.pose_r[-1]))/(norm**2) * (matrix@matrix)
            matrix = matrix.transpose(1, 0)

        return grid @ matrix + self.pose_t

    def forward(self, grid):
        # assume every value in the grid is between -1 and 1
        grid = self.translate(grid).unsqueeze(0)
        outputs = F.grid_sample(self.voxel, grid, mode=self.mode,
                                padding_mode='reflection', align_corners=True)
        return outputs[0].permute(1, 0, 2, 3)


class VoxelNeRV(nn.Module):
    def __init__(self, video, channels: list, upscale_factors=[5, 4, 4],
                 voxel_resolution=None, activation='gelu', kernel_size=3,
                 bias=True, n_voxel=4, block_type='basic'):
        super().__init__()
        self.resolution = video.shape
        total_upscale = np.prod(upscale_factors)
        self.resolution_low = [d//total_upscale for d in self.resolution[-2:]]

        if len(channels) == 1:
            channels = channels * (len(upscale_factors) + 1)

        if voxel_resolution is None:
            voxel_resolution = torch.tensor(
                [int(np.ceil(self.resolution[0] / time_upscale)),
                 *self.resolution_low])

        self.voxels = nn.ModuleList(
            [Voxel(voxel_resolution, channels[0] // n_voxel)
             for _ in range(n_voxel)])

        block = BasicBlock if block_type == 'basic' else NeRVBlock
        self.body = nn.Sequential(
            *[block(channels[i], channels[i+1], kernel_size, u, bias,
                    activation)
              for i, u in enumerate(upscale_factors)])

        self.output_layer = nn.Conv2d(channels[-1], 3, 
                                      kernel_size, 1, kernel_size//2, bias=bias)

    def forward(self, inputs):
        outputs = torch.cat([v(inputs) for v in self.voxels], 1)
        outputs = self.body(outputs)
        return torch.sigmoid(self.output_layer(outputs))


class NeRV(nn.Module):
    def __init__(self, input_dim, output_resolution: list, channels: int,
                 pos_enc_freqs=40, mlp_hidden_dim: list = [512],
                 activation='swish', upscale_factors=[5, 4, 4],
                 kernel_size=5, bias=True):
        super().__init__()

        total_upscale = np.prod(upscale_factors)
        assert output_resolution[-2] % total_upscale == 0
        assert output_resolution[-1] % total_upscale == 0

        self.input_resolution = [d // total_upscale
                                 for d in output_resolution[-2:]]

        self.pos_enc = PosEncoding(input_dim[-1], pos_enc_freqs)
        self.pos_enc_real = PosEncoding(
            2, np.ceil(np.log2([r*8 for r in self.input_resolution])).astype('int')+1)

        mlp_dim_list = [self.pos_enc.get_output_size(), *mlp_hidden_dim,
                        np.prod(self.input_resolution)*2] # channels]
        self.stem = MLP(mlp_dim_list, activation=activation)

        self.body = []
        for i, u in enumerate(upscale_factors):
            self.body.append(NeRVBlock(channels if i else self.pos_enc_real.get_output_size(),
                                       channels, kernel_size,
                                       u, bias, activation))
        self.body = nn.Sequential(*self.body)

        self.output_layer = nn.Conv2d(channels, output_resolution[-3],
                                      kernel_size, 1, kernel_size//2, bias=bias)

    def forward(self, coords):
        outputs = self.pos_enc(coords.unsqueeze(0))
        outputs = self.stem(outputs)
        outputs = outputs.view(outputs.size(0), -1, *self.input_resolution)

        outputs = outputs.permute(0, 2, 3, 1) # [T, H, W, C]
        outputs = make_grid(*self.input_resolution).unsqueeze(0).to(outputs.device) + outputs
        outputs = self.pos_enc_real(outputs)
        outputs = outputs.permute(0, 3, 1, 2) # [T, C, H, W]

        outputs = self.body(outputs)
        outputs = self.output_layer(outputs)
        outputs = torch.sigmoid(outputs)

        return outputs

