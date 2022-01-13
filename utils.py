import numpy as np
import os
import skvideo.io
import torch
import torchvision


def load_video(name, scale=1, start_frame=0, num_frames=None):
    name, ext = os.path.splitext(name)
    if ext == '':
        ext = '.avi'

    frames = skvideo.io.vread(f'{name}{ext}')
    if num_frames is None:
        num_frames = len(frames)
    frames = frames[start_frame:start_frame+num_frames]
    frames = torch.from_numpy(frames).float() / 255. # [0, 255] to [0, 1]
    frames = frames.permute(0, 3, 1, 2) # to [time, chan, height, width]

    if scale != 1:
        frames = torchvision.transforms.functional.resize(
            frames,
            tuple(int(s*scale) for s in frames.size()[-2:]), # new video size
            antialias=True)

    return frames


def make_input_grid(*input_size, minvalue=-1, maxvalue=1):
    # generate grid for a given input size
    # 2D coordinates: use make_input_grid(H, W) -> (H, W, 2) shaped tensor
    # 3D coordinates: use make_input_grid(T, H, W) -> (T, H, W, 3) shaped tensor
    return torch.stack(torch.meshgrid(*[torch.linspace(minvalue, maxvalue, S)
                                        for S in input_size]), -1)


def make_flow_grid(H, W):
    # generates (H, W, 2) shaped tensor
    flow_grid = torch.stack(torch.meshgrid(torch.arange(0, H),
                                           torch.arange(0, W)), -1).float()
    return torch.flip(flow_grid, (-1,)) # from (y, x) to (x, y)


def apply_flow(prev_coords, flow, H=None, W=None, normalize=True):
    # assume flow_grid and pred_flow are pixel locations
    next_coords = prev_coords.to(flow.device) + flow
    if normalize:
        # normalize to [-1, 1]
        assert H is not None and W is not None, 'both H,W must not be None'
        next_coords = 2 * next_coords \
                    / torch.tensor([[[[W-1, H-1]]]]).to(next_coords.device) - 1 
    return next_coords


def bilinear_interpolation(images, coords):
    '''
    inputs
        images(float): [N, C, H, W]
        coords(float): [*, 3] (image_idx, x, y)
    '''
    xy = torch.floor(
        torch.tensor([[0., 0.], [0, 1], [1, 0], [1, 1]]).to(coords.device) \
        + coords[..., None, 1:])

    xy = torch.clamp(
        xy,
        min=torch.tensor([0., 0.]).to(xy.device),
        max=torch.tensor(images.size()[-1:-3:-1]).float().to(xy.device) - 1)

    # [*, 4, C] shaped values from images
    values = images[coords[..., None, 0].long(), :,
                    xy[..., 1].long(), xy[..., 0].long()]

    weights = 1 - torch.abs(xy - coords[..., None, 1:])
    weights = weights.prod(-1, keepdim=True)

    return torch.sum(weights * values, -2)

