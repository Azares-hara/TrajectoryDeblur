import torch
from torch import nn
import torch.nn.functional as F
import piq
from lossfunc.lpipsregistry import get_lpips

def _expand(img):
    """Ensure tensor is 4D [B, C, H, W]."""
    while img.ndim < 4:
        img = img.unsqueeze(0)
    return img

def _to_01(img):
    """
    Convert image to [0, 1] range.
    """
    if img.min() < 0.0:
        return (img + 1.0) / 2.0
    return img.clamp(0.0, 1.0)

def safe_fft_loss(x, y):
    fft_x = torch.fft.fft2(x, norm="ortho")
    fft_y = torch.fft.fft2(y, norm="ortho")
    mag_x = torch.nan_to_num(torch.abs(fft_x), nan=0.0, posinf=1e4, neginf=0.0)
    mag_y = torch.nan_to_num(torch.abs(fft_y), nan=0.0, posinf=1e4, neginf=0.0)
    return torch.mean(torch.abs(mag_x - mag_y))


class LPIPSMetric(nn.Module):
    def __init__(self, net='squeeze', device='cuda:0', use_half=False):
        super().__init__()
        self.lpips_fn = get_lpips(net=net, device=device, use_half=use_half)

    def forward(self, im1, im2):
        val = self.lpips_fn(im1, im2)
        return torch.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)

class PSNR(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device

    def forward(self, im1, im2, data_range=None):
        im1 = _to_01(im1.to(self.device))
        im2 = _to_01(im2.to(self.device))
        data_range = 1.0 if data_range is None else data_range
        mse  = ((im1 - im2) ** 2).mean().clamp(min=1e-8)
        psnr = 10.0 * torch.log10((data_range ** 2) / mse)
        return psnr
        
class SSIM(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super().__init__()
        self.device_type = device_type
        self.dtype       = dtype

        truncate  = 3.5
        sigma     = 1.5
        r         = int(truncate * sigma + 0.5)
        win_size  = 2 * r + 1
        col = torch.tensor(
            [-(x - win_size // 2) ** 2 / (2 * sigma ** 2) for x in range(win_size)]
        ).exp().unsqueeze(1)
        weight = col.mm(col.t())
        weight = weight / weight.sum()
        self.register_buffer(
            'weight_1ch',
            weight.unsqueeze(0).unsqueeze(0).to(dtype=self.dtype)
        )

    def forward(self, im1, im2, data_range=None):
        device = torch.device(self.device_type)
        with torch.no_grad():
            im1 = _to_01(im1.to(device, dtype=self.dtype, non_blocking=True))
            im2 = _to_01(im2.to(device, dtype=self.dtype, non_blocking=True))
            im1 = _expand(im1)
            im2 = _expand(im2)

            nch = im1.shape[1]
            w = self.weight_1ch.to(device).expand(nch, 1, -1, -1)

            def filt(x):
                return F.conv2d(x, w, padding=w.shape[-1] // 2, groups=nch)

            K1, K2 = 0.01, 0.03
            R       = 1.0 if data_range is None else data_range
            C1, C2  = (K1 * R) ** 2, (K2 * R) ** 2

            ux  = filt(im1)
            uy  = filt(im2)
            uxx = filt(im1 * im1)
            uyy = filt(im2 * im2)
            uxy = filt(im1 * im2)

            vx  = uxx - ux * ux
            vy  = uyy - uy * uy
            vxy = uxy - ux * uy

            S = ((2.0 * ux * uy + C1) * (2.0 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
            return S.mean()
class NIQEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im):
        im = _to_01(im)
        return piq.niqe(im).mean()


class BRISQUEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im):
        im = _to_01(im)
        return piq.brisque(im).mean()

def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channels=3):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = channels
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.register_buffer('window', self._create_window(window_size, channels))

    def _create_window(self, window_size, channels):
        _1D = gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        #img1/img2 assumed in [0,1] or [-1,1]; normalize window to device/dtype
        window = self.window.to(img1.device, dtype=img1.dtype)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channels)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu12
        ssim_map = ((2 * mu12 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return 1.0 - ssim_map.mean() if self.size_average else 1.0 - ssim_map