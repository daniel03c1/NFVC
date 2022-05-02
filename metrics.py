import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


class PSNR:
    def __call__(self, inputs, targets):
        return -10 * torch.log10(F.mse_loss(inputs.detach(), targets.detach()))


class SSIM:
    def __call__(self, inputs, targets):
        # return ms_ssim(inputs.detach(), targets.detach(), data_range=1,
        #                size_average=True)
        if min(*inputs.shape[-2:]) < 160:
            return 0
        return ms_ssim(inputs.detach(), targets.detach(), data_range=1,
                       size_average=True)

