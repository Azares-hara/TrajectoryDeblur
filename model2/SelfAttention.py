import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv_block(in_c, out_c, kernel=3, isPad=True, isRelu=True, norm="instance"):
    layers = []
    if isPad:
        layers.append(nn.ReflectionPad2d(1))
    layers.append(nn.Conv2d(in_c, out_c, kernel, bias=False))
    if norm == "instance":
        layers.append(nn.GroupNorm(1, out_c, affine=True))
    elif norm == "batch":
        layers.append(nn.BatchNorm2d(out_c))
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm type: '{norm}'. Expected 'instance', 'batch', or 'none'.")
    if isRelu:
        layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)

def normalize_score(score):
    return torch.sigmoid(score)

def frequency_modulate(feat):
    feat = feat.float()
    fft_feat = torch.fft.fft2(feat, norm="ortho")
    fft_real = torch.nan_to_num(fft_feat.real, nan=0.0, posinf=1e4, neginf=-1e4)
    fft_imag = torch.nan_to_num(fft_feat.imag, nan=0.0, posinf=1e4, neginf=-1e4)
    fft_feat = torch.complex(fft_real, fft_imag)
    fft_feat = torch.fft.fftshift(fft_feat, dim=(-2, -1))

    h, w = fft_feat.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=feat.device),
        torch.arange(w, device=feat.device),
        indexing="ij",
    )
    center_y, center_x = h // 2, w // 2
    dist  = torch.sqrt((yy - center_y) ** 2.0 + (xx - center_x) ** 2.0)
    eps   = 1e-8
    sigma = 0.1 * min(h, w) + eps          
    mask  = 1.0 - torch.exp(-(dist / sigma) ** 2)
    mask  = mask.unsqueeze(0).unsqueeze(0)

    fft_mod = fft_feat * mask
    fft_mod = torch.fft.ifftshift(fft_mod, dim=(-2, -1))
    out     = torch.real(torch.fft.ifft2(fft_mod, norm="ortho"))
    return out


def freq_energy(feat):
    fft_feat  = torch.fft.fft2(feat, norm="ortho")
    fft_real  = torch.nan_to_num(fft_feat.real, nan=0.0, posinf=1e4, neginf=-1e4)
    fft_imag  = torch.nan_to_num(fft_feat.imag, nan=0.0, posinf=1e4, neginf=-1e4)
    fft_feat  = torch.complex(fft_real, fft_imag)
    mag       = torch.abs(fft_feat)
    dc        = mag[:, :, :1, :1]
    high_freq = mag - dc
    return high_freq.mean(dim=(1, 2, 3))


def normalize_score(score):
    min_val = score.min().detach()
    max_val = score.max().detach()
    denom   = (max_val - min_val).clamp(min=1e-8)
    return (score - min_val) / denom


def blur_loss(blur_prob, feat):
    blur_prob  = torch.nan_to_num(blur_prob, nan=0.0)
    blur_prob  = torch.clamp(blur_prob, 0.0, 1.0)
    score      = freq_energy(feat)
    norm_score = normalize_score(score)
    target     = 1.0 - norm_score
    return F.l1_loss(blur_prob.mean(dim=(1, 2, 3)), target)

class ResidualRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.net(x) + x)

class EdgeAwareAttention(nn.Module):
    def __init__(self, in_c=64, num_feat=32):
        super().__init__()
        self.feat_conv = nn.Sequential(
            conv_block(in_c, num_feat),
            conv_block(num_feat, num_feat),
        )
        self.edge_conv = nn.Sequential(
            conv_block(num_feat, num_feat),
            nn.Conv2d(num_feat, 1, kernel_size=1),
        )
        self.attn_conv  = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.refine     = ResidualRefine(num_feat)
        self.freq_alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.edge_scale = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(self, x):
        feat     = self.feat_conv(x)
        edge_map = torch.sigmoid(self.edge_conv(feat)) * torch.clamp(self.edge_scale, 0.1, 2.0)
        out      = self.attn_conv(feat * edge_map)
        out      = self.refine(out)
        return {"out": out, "edge_map": edge_map}

class Self_Attn_FM(nn.Module):
    """Spatial self-attention with optional pooling for memory efficiency."""
    def __init__(self, in_dim, pool_factor=4):
        super().__init__()
        self.query    = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key      = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value    = nn.Conv2d(in_dim, in_dim,      1)
        self.out_proj = nn.Conv2d(in_dim, in_dim,      1)
        self.gamma    = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.pool_factor = pool_factor
        self._key_dim    = in_dim // 8

    def forward(self, x):
        B, C, H, W = x.size()
        xp = F.avg_pool2d(x, self.pool_factor, self.pool_factor) if self.pool_factor > 1 else x
        Hp, Wp = xp.shape[2], xp.shape[3]

        Q = self.query(xp).view(B, -1, Hp * Wp)    
        K = self.key(xp).view(B, -1, Hp * Wp)   
        V = self.value(xp).view(B, -1, Hp * Wp)    

        scale  = math.sqrt(self._key_dim)
        energy = torch.bmm(Q.permute(0, 2, 1), K) / scale  
        attn   = F.softmax(energy, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1))  
        out = out.view(B, C, Hp, Wp)
        if self.pool_factor > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        out = self.out_proj(out)
        return self.gamma * out + x, attn
