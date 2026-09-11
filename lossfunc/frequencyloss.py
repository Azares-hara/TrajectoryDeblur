import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_gaussian_kernel
import math

class FrequencyLoss(nn.Module):

    def __init__(self, weight=1.0, use_phase=True, multi_scale=True,
                 scales=None, return_components=False,
                 kernel_size=5, sigma=1.0):
        super().__init__()
        self.weight = weight
        self.use_phase = use_phase
        self.multi_scale = multi_scale
        self.scales = scales if scales is not None else (2, 4)
        self.return_components = return_components
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = get_gaussian_kernel(kernel_size, sigma) 
        self.register_buffer('kernel', kernel)

    def _apply_blur(self, img):
        C = img.size(1)
        kernel = self.kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(img, kernel, padding=self.kernel_size // 2, groups=C)

    @staticmethod
    def _safe_fft2(x):
        fft = torch.fft.fft2(x, norm="ortho")
        return torch.complex(
            torch.nan_to_num(fft.real, nan=0.0, posinf=1e4, neginf=-1e4),
            torch.nan_to_num(fft.imag, nan=0.0, posinf=1e4, neginf=-1e4),
        )

    def forward(self, restored_image, blurred_image):
        reblurred = self._apply_blur(restored_image)

        r_fft = self._safe_fft2(reblurred)
        b_fft = self._safe_fft2(blurred_image)

        num_pixels = blurred_image.numel()

        #magnitude loss
        mag_loss  = torch.sum(torch.abs(torch.abs(r_fft) - torch.abs(b_fft))) / num_pixels
        freq_loss = mag_loss

        phase_loss = torch.tensor(0.0, device=restored_image.device)
        if self.use_phase:
            phase_diff = torch.abs(torch.angle(r_fft) - torch.angle(b_fft))
            
            phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
            phase_loss = torch.mean(phase_diff)
            freq_loss = freq_loss + 0.5 * phase_loss
        ms_loss = torch.tensor(0.0, device=restored_image.device)
        if self.multi_scale:
            for scale in self.scales:
                r_down = F.avg_pool2d(reblurred,scale)
                b_down = F.avg_pool2d(blurred_image,scale)
                r_fft_s = torch.abs(self._safe_fft2(r_down))
                b_fft_s = torch.abs(self._safe_fft2(b_down))
                ms_loss = ms_loss + torch.mean(torch.abs(r_fft_s - b_fft_s))
            freq_loss = freq_loss + ms_loss

        total_loss = self.weight * freq_loss

        if self.return_components:
            return {
                "total_loss": total_loss,
                "mag_loss":mag_loss,
                "phase_loss":phase_loss,
                "multi_scale_loss": ms_loss,
            }
        return total_loss