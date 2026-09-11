import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils import match_size


class PatchDiscriminator(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3):
        super().__init__()

        def conv(in_c, out_c, stride=1, pad=None):
            if pad is None:
                pad = (kernel_size - 1) // 2
            return spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size,
                          stride=stride, padding=pad, bias=False)
            )
        self.blocks = nn.ModuleList([
            conv(3, n_feats, 1), nn.LeakyReLU(0.2, inplace=False),
            conv(n_feats,n_feats * 2, 2), nn.InstanceNorm2d(n_feats * 2), nn.LeakyReLU(0.2, inplace=False),
            conv(n_feats * 2, n_feats * 4, 2), nn.InstanceNorm2d(n_feats * 4), nn.LeakyReLU(0.2, inplace=False),
            conv(n_feats * 4, n_feats * 8, 2), nn.InstanceNorm2d(n_feats * 8), nn.LeakyReLU(0.2, inplace=False),
        ])
        self.score_conv = conv(n_feats * 8, 1, 1)
        self._feat_indices = {1, 4, 7, 10}

    def forward(self, x):
        feats = []
        out = x
        for i, layer in enumerate(self.blocks):
            out = layer(out)
            if i in self._feat_indices:
                feats.append(out)
        score = self.score_conv(out)
        return score, feats


class FrequencyDiscriminator(nn.Module):
    def __init__(self, n_feats=64, downsample_size=None):
        super().__init__()
        self.downsample_size = downsample_size

        # FIXED: Input is now 9 channels (3 normalized mag + 3 phase + 3 global energy)
        self.layers = nn.ModuleList([
            spectral_norm(nn.Conv2d(9, n_feats, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(n_feats, n_feats * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(n_feats * 2, 1, 3, padding=1)),
        ])
        self._feat_indices = {1, 3}

    def forward(self, x):
        inp = x

        if self.downsample_size is not None:
            inp = F.interpolate(
                inp,
                size=(self.downsample_size, self.downsample_size),
                mode="bilinear",
                align_corners=False,
            )

        fft_complex = torch.fft.fft2(inp, norm="ortho")
        fft_complex = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        fft_complex = torch.complex(
            torch.nan_to_num(fft_complex.real, nan=0.0, posinf=1e4, neginf=-1e4),
            torch.nan_to_num(fft_complex.imag, nan=0.0, posinf=1e4, neginf=-1e4),
        )

        fft_mag_raw = torch.log1p(torch.abs(fft_complex))
        fft_phase = torch.angle(fft_complex)

        # Per-sample normalization (preserves shape, loses absolute energy)
        fft_mag_norm = (fft_mag_raw - fft_mag_raw.mean(dim=(-2, -1), keepdim=True)) / (
            fft_mag_raw.std(dim=(-2, -1), keepdim=True) + 1e-8
        )
        fft_phase = fft_phase / (torch.pi + 1e-8)

        # NEW: Global energy statistic — broadcasted mean of raw magnitude
        # This preserves the absolute energy difference between blur and sharp
        fft_global = fft_mag_raw.mean(dim=(-2, -1), keepdim=True).expand_as(fft_mag_raw)

        fft_input = torch.cat([fft_mag_norm, fft_phase, fft_global], dim=1)
        fft_input = torch.nan_to_num(fft_input, nan=0.0, posinf=1.0, neginf=-1.0)

        feats = []
        out = fft_input
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out)
            if i in self._feat_indices:
                feats.append(out)
        score = self.layers[-1](out)
        return score, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.patch_disc_full = PatchDiscriminator(n_feats=n_feats)
        self.patch_disc_half = PatchDiscriminator(n_feats=n_feats)
        self.patch_disc_quarter = PatchDiscriminator(n_feats=n_feats)
        self.freq_disc = FrequencyDiscriminator(n_feats=n_feats)

    def forward(self, x):
        score_full, feats_full = self.patch_disc_full(x)
        score_half,feats_half = self.patch_disc_half(
            F.interpolate(x, scale_factor=0.5,  mode="bilinear", align_corners=False)
        )
        score_quarter, feats_quarter = self.patch_disc_quarter(
            F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        )
        score_freq,feats_freq = self.freq_disc(x)

        return {
            "score_full": score_full,
            "score_half": score_half,
            "score_quarter":score_quarter,
            "score_freq": score_freq,
            "features": feats_full + feats_half + feats_quarter + feats_freq,
        }

    def feature_matching_loss(self, real_feats, fake_feats):
        loss = 0.0
        n = min(len(real_feats), len(fake_feats))
        for f, r in zip(fake_feats[:n], real_feats[:n]):
            f = match_size(f, r)
            if f.shape[1] != r.shape[1]:
                min_c = min(f.shape[1], r.shape[1])
                warnings.warn(
                    f"feature_matching_loss: channel mismatch {f.shape[1]} vs "
                    f"{r.shape[1]}, truncating to {min_c}.",
                    stacklevel=2,
                )
                f = f[:, :min_c]
                r = r[:, :min_c]
            loss += F.l1_loss(
                torch.nan_to_num(f,nan=0.0),
                torch.nan_to_num(r.detach(), nan=0.0),
            )
        return loss
