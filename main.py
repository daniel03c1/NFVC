import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm
from torch import optim

from embedding import *
from metrics import PSNR, SSIM, LPIPS, MSE
from network import *
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# Configure
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--name', type=str, required=True,
                    help='the name of the model')

# Model
parser.add_argument('--video-name', type=str,
                    default='../neural_vic_temp/data/YachtRide',
                    help='video name')
parser.add_argument('--in_features', default=3, type=int, choices=[3, 4])
parser.add_argument('--rgb_only', action='store_true')
parser.add_argument('--hidden_features', type=int, default=72,
                    help='the number of hidden units (default: 72)')
parser.add_argument('--n_hidden_layers', type=int, default=3,
                    help='the number of layers (default: 3)')
parser.add_argument('--emb_size', type=int, default=72,
                    help='feature dimension (default: 72)')
parser.add_argument('--activation', default='ReLU', type=str,
                    help='activation function')
parser.add_argument('--sep_model', action='store_true')

# Training
parser.add_argument('--seed', type=int, default=50236,
                    help='random seed (default: 50236)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=10000)

# custom parameters
parser.add_argument('--scale', type=float, default=0.5,
                    help='image resolution scale')
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--n_frames', type=int, default=None)
parser.add_argument('--keyframe_interval', type=int, default=None)
parser.add_argument('--qf', type=int, default=1,
                    help='jpeg quality factor (default: 1)')


def generate_indices(n_frames, keyframe_interval=None):
    if keyframe_interval is None:
        keyframe_interval = n_frames
    n_chunks = int(np.round(n_frames / float(keyframe_interval)))
    chunk_indices = torch.round(torch.linspace(0, n_frames, n_chunks+1)).long()

    keyframe_indices = ((chunk_indices[:-1] + chunk_indices[1:])/2).long()

    return keyframe_indices, chunk_indices


def extract_keyframes(frames, keyframe_indices, quality_factor=95):
    IMG_NAME = f'{np.random.randint(10000):04d}.jpeg'

    keyframes = []
    keyframes_nbyte = 0

    for i in keyframe_indices:
        keyframe = (target_frames[i] * 255).permute(1, 2, 0)
        keyframe = keyframe.numpy().astype('uint8')
        cv2.imwrite(IMG_NAME, keyframe[..., ::-1], # RGB to BGR
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])

        keyframes_nbyte += os.stat(IMG_NAME).st_size
        keyframe = cv2.imread(IMG_NAME)
        keyframe = torch.from_numpy(keyframe / 255.).float().permute(2, 0, 1)
        keyframe = torch.flip(keyframe, [-3]) # BGR to RGB
        keyframes.append(keyframe)

    os.remove(IMG_NAME)

    return torch.stack(keyframes, 0), keyframes_nbyte


if __name__=='__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    metrics = [PSNR(), SSIM(), LPIPS()]

    '''     DATA     '''
    # 1. data loading
    target_frames = load_video(args.video_name, args.scale,
                               args.start_frame, args.n_frames)

    args.n_frames = len(target_frames)
    if args.keyframe_interval is None:
        args.keyframe_interval = len(target_frames)

    # 2. generate keyframes
    H, W = target_frames.shape[-2:]

    keyframe_indices, chunk_indices = \
        generate_indices(args.n_frames, args.keyframe_interval)
    n_keyframes = len(keyframe_indices)
    width = max(chunk_indices[1:]-chunk_indices[:-1])

    if args.rgb_only:
        keyframe_nbyte = 0
    else:
        flow_grid = make_flow_grid(H, W)
        keyframe_mapper = (torch.arange(args.n_frames)[None, :]
                           >= chunk_indices[:, None]).sum(0) - 1
        keyframes, keyframe_nbyte = extract_keyframes(
            target_frames, keyframe_indices, args.qf)
        keyframes = keyframes.cuda()

    # 3. generate input_grid
    if not args.sep_model or len(keyframe_indices) == 1:
        # single model
        args.sep_model = False

        if args.in_features == 3 or n_keyframes == 1:
            args.in_features = 3
            input_grid = make_input_grid(len(target_frames), H, W)
            grid_mapper = torch.arange(args.n_frames)
        else: # args.in_features == 4
            input_grid = make_input_grid(
                len(keyframe_indices), width, H, W)
            grid_mapper = np.stack(
                [keyframe_mapper,
                 torch.arange(args.n_frames)-chunk_indices[keyframe_mapper]],
                -1)
    else:
        # multiple model
        args.sep_model = True
        input_grid = make_input_grid(width, H, W)
        grid_mapper = (torch.arange(args.n_frames)
                       - chunk_indices[keyframe_mapper])

    '''     MODEL     '''
    args.out_features = 3 + 2 * (not args.rgb_only)

    nets = [
        NeuralFieldsNetwork(args.in_features, args.out_features,
                            args.hidden_features, args.n_hidden_layers,
                            RFF(args.in_features, args.emb_size//2, np.pi))
        for i in range(1 + args.sep_model * (n_keyframes-1))
    ]
    nets = [nn.DataParallel(net.cuda()) for net in nets]

    optimizers = [
        optim.Adam(nets[i].parameters(), lr=args.lr, weight_decay=1e-5)
        for i in range(len(nets))]
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max=args.epochs)
        for i in range(len(nets))]

    '''     CONFIG & LOGS     '''
    dirname = "./result/{}_{}/".format(
        os.path.splitext(os.path.split(args.video_name)[-1])[0], args.name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # save config
    with open(os.path.join(dirname, 'config.txt'), 'w') as o:
        o.write(str(vars(args)))

    print(args)
    print(f'Image size: ({H}, {W})')
    print(f'# of frames: {len(target_frames)}')
    n_params = sum([p.numel() for net in nets for p in net.parameters()])
    print(f'n_params: {n_params}, approx size: {n_params*4 + keyframe_nbyte}')

    '''     TRAINING     '''
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = H * W
    n_steps = (args.n_frames * H * W) // batch_size

    logs = []

    with tqdm.tqdm(range(args.epochs)) as loop:
        for epoch in loop:
            for net in nets:
                net.train()

            for i in range(n_steps):
                x = torch.randint(W, (batch_size,))
                y = torch.randint(H, (batch_size,))
                if not args.sep_model:
                    n = 0
                    j = torch.randint(args.n_frames, (batch_size,))
                else: # args.sep_model:
                    n = torch.randint(n_keyframes, ())
                    j = torch.randint(chunk_indices[n], chunk_indices[n+1],
                                      (batch_size,))

                targets = target_frames[j, :, y, x].cuda()
                inputs = input_grid[grid_mapper[j], y, x].cuda()
                outputs = nets[n](inputs)

                if args.rgb_only:
                    outputs = torch.sigmoid(outputs) # to RGB
                else: # our approach
                    pred_flow, outputs = outputs[..., :2], outputs[..., 2:]
                    outputs = torch.tanh(outputs)

                    warped = bilinear_interpolation(
                        keyframes,
                        torch.stack([keyframe_mapper[j].cuda(),
                                     x.float().cuda() + pred_flow[..., 0],
                                     y.float().cuda() + pred_flow[..., 1]],
                                    -1))
                    outputs = (warped + outputs).clamp(0, 1)

                loss = F.l1_loss(outputs, targets)

                optimizers[n].zero_grad()
                loss.backward()
                optimizers[n].step()

            for scheduler in schedulers:
                scheduler.step()

            # logging
            if (epoch+1) % args.eval_interval == 0:
                for net in nets:
                    net.eval()

                epoch_log = np.zeros(len(metrics))

                for i in range(args.n_frames):
                    preds = net(input_grid[i].unsqueeze(0).cuda())
                    true_rgb = target_frames[i].unsqueeze(0).cuda()

                    if args.rgb_only:
                        pred_rgb = torch.sigmoid(preds) # to RGB
                        pred_rgb = pred_rgb.permute(0, 3, 1, 2) # to [1, C, H, W]
                    else: # our approach
                        pred_flow, residuals = preds[..., :2], preds[..., 2:]
                        residual = residual.permute(2, 0, 1) # to [3, H, W]


                        warped = bilinear_interpolation(
                            keyframes[keyframe_mapper[i]].unsqueeze(0).cuda(),
                            torch.stack([keyframe_mapper[j].cuda(),
                                         x.float().cuda() + pred_flow[..., 0],
                                         y.float().cuda() + pred_flow[..., 1]],
                                        -1))

                        pred_rgb = (warped + residual).clamp(0, 1)

                    for i in range(len(metrics)):
                        epoch_log[i] += \
                            metrics[i](pred_rgb, true_rgb).item() / args.n_frames

                for net in nets:
                    net.train()

                loop.set_postfix({str(m): epoch_log[i]
                                  for i, m in enumerate(metrics)})
                logs.append(epoch_log)

                # leave a log
                np.savetxt(os.path.join(dirname, "metrics.csv"),
                           logs, delimiter=",")

