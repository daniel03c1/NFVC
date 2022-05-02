import argparse
import glob
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import *
from metrics import *
from video_utils import *


parser = argparse.ArgumentParser()

# dataset parameters
parser.add_argument('--video', required=True, type=str,
                    help='the path for the video')
parser.add_argument('--name', required=True)
parser.add_argument('--out_folder', default='compressed',
                    help='folder to output images and model checkpoints')
parser.add_argument('--resume', action='store_true',
                    help='whether to continue from the saved checkpoint')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite the output dir if already exists')

# architecture parameters
parser.add_argument('--mlp_hidden_dim', type=int, nargs='+', default=[512])
parser.add_argument('--channels', type=int, nargs='+', default=[42])
parser.add_argument('--upscale_factors', type=int, nargs='+',
                    default=[5, 4, 4])
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--activation', type=str, default='gelu')

parser.add_argument('--n_voxel', type=int, default=4)
parser.add_argument('--block_type', type=str, default='basic')

# General training setups
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--grad_clip', type=float, default=1.)

# evaluation parameters
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--eval_freq', type=int, default=50)


def train(args):
    # 1. Video
    video = load_video(args.video).cuda()
    if 'uvg' in args.video:
        video = video[:100]
    print(video.shape)

    # 2. Model
    metrics = [PSNR(), SSIM()]
    start_epoch = 0
    n_freqs = int(np.ceil(np.log2(len(video)))) + 1

    voxel_shape = torch.tensor(video.shape)[torch.tensor([0, 2, 3])] // 64
    voxel_shape = voxel_shape.clamp(min=5)
    print(voxel_shape)
    model = VoxelNeRV(video, args.channels, args.upscale_factors, voxel_shape,
                      activation=args.activation, kernel_size=args.kernel_size,
                      n_voxel=args.n_voxel, block_type=args.block_type)
    model = torch.jit.script(model)

    total_bytes = 2 * sum([p.numel() for p in model.parameters()])
    bpp = 8 * total_bytes / len(video) / np.prod(video.shape[-2:])
    print(f'Model Params: {total_bytes/1e6:.4f}MB (bpp: {bpp:.4f})')

    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (args.epochs-start_epoch))
    scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        raise NotImplemented("not yet...")

    model = nn.DataParallel(model).cuda()

    total_upscale = np.prod(args.upscale_factors)
    grids = make_grid(len(video), *[v//total_upscale for v in video.shape[-2:]])
    grids = grids.cuda()

    # 3. Train
    model.train()
    best_score = 0

    for epoch in tqdm.tqdm(range(start_epoch, args.epochs)):
        # iterate over dataloader
        for i in torch.randperm(len(video)).cuda().split(args.batch_size):
            frame = torch.index_select(video, 0, i).half() # [B, C, H, W]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                inputs = torch.index_select(grids, 0, i) # [B, H, W, 3]
                outputs = model(inputs)

                # loss = 0.7 * F.mse_loss(outputs, frame) \
                #      + 0.3 * (1 - ssim(outputs, frame))
                loss = F.mse_loss(outputs, frame)
                assert not torch.isnan(loss)
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            scores = evaluate(model, grids, video, metrics)
            print(scores)

            if best_score < scores[0]:
                best_score = scores[0]
                checkpoint = {'epoch': epoch+1,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f'{args.out_folder}/best_score.pth')

    checkpoint = {'epoch': epoch+1, 'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, f'{args.out_folder}/latest.pth')


@torch.no_grad()
def evaluate(model, grids, video, metrics):
    n_frames = len(video)
    results = [0] * len(metrics)

    model.eval()

    for i in range(n_frames):
        targets = video[i:i+1]

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(torch.index_select(grids, 0, torch.ones((1,)).long().cuda()*i))

        targets = F.adaptive_avg_pool2d(targets, outputs.shape[-2:])
        for j, m in enumerate(metrics):
            results[j] += m(outputs.float(), targets) / n_frames            

    model.train()

    return results


if __name__ == '__main__':
    args = parser.parse_args()

    print(vars(args))
    torch.set_printoptions(precision=3)

    args.out_folder = os.path.join(args.out_folder, args.name)

    if args.overwrite and os.path.isdir(args.out_folder):
        shutil.rmtree(args.out_folder)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    with open(os.path.join(args.out_folder, 'config.cfg'), 'w') as o:
        o.write(str(vars(args)))

    train(args)

