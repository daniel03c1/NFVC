import PIL
import glob
import os
import skvideo.io
import torch
import torchvision
import torchvision.transforms as transforms


def load_video(path, *args, **kwargs):
    if os.path.isdir(path):
        return load_video_from_images(path, *args, **kwargs)
    elif os.path.splitext(path)[1].lower() in ['.avi', '.mp4']:
        return load_video_from_video(path, *args, **kwargs)
    else:
        raise ValueError('Path must be one of a directory containing '
                         'PNG files or a ".avi" formatted video')


def load_video_from_images(image_folder, start_frame=None, n_frames=None,
                           scale=1., antialias=False):
    frames = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    if start_frame is None:
        start_frame = 0
    if n_frames is None:
        n_frames = len(frames)

    frames = [transforms.functional.to_tensor(PIL.Image.open(f))
              for f in frames[start_frame:start_frame+n_frames]]
    frames = torch.stack(frames, 0)

    if scale != 1:
        frames = torchvision.transforms.functional.resize(
            frames,
            tuple(int(s*scale) for s in frames.size()[-2:]),
            antialias=antialias)

    return frames


def load_video_from_video(video_path, start_frame=None, n_frames=None,
                          scale=1., antialias=False):
    name, ext = os.path.splitext(video_path)

    frames = skvideo.io.vread(f'{name}{ext}')
    if start_frame is None:
        start_frame = 0
    if n_frames is None:
        n_frames = len(frames)
    frames = frames[start_frame:start_frame+n_frames]
    frames = torch.from_numpy(frames).float() / 255. # [0, 255] to [0, 1]
    frames = frames.permute(0, 3, 1, 2) # to [time, chan, height, width]

    if scale != 1:
        frames = torchvision.transforms.functional.resize(
            frames,
            tuple(int(s*scale) for s in frames.size()[-2:]), # new video size
            antialias=antialias)

    return frames

