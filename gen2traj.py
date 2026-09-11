%%writefile /kaggle/working/Deblur/model2/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model2.SelfAttention import EdgeAwareAttention


class BlurPoolDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.blur = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1,
                              padding=kernel_size // 2, groups=in_channels, bias=False)
        with torch.no_grad():
            k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
            k = k / k.sum()
            self.blur.weight.copy_(k.expand(in_channels, 1, 3, 3))
        self.blur.weight.requires_grad = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(1, out_channels, affine=True)
        
    def forward(self, x):
        x = self.blur(x)
        return F.relu(self.norm(self.conv(x)))

class TrajectoryHead(nn.Module):
    def __init__(self, in_channels=64, base_channels=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(1, base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(1, base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(1, base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(1, base_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.traj_gate = nn.Conv2d(base_channels, 1, 1)
        with torch.no_grad():
            self.traj_gate.bias.fill_(-1.0)  # weaker suppression: sigmoid(-1) ˜ 0.27
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        gate = torch.sigmoid(self.traj_gate(d2))
        return self.out(d2 + gate * e1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels, affine=True, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels, affine=True, eps=1e-5)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm_up = nn.GroupNorm(1, out_channels, affine=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.relu(self.norm_up(self.up(x)))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResUNetGenerator(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3,
                 base_channels=64, use_sa=False, out_activation="tanh"):
        super().__init__()
        self.num_feat = args.num_feat
        self.out_activation = out_activation
        self.use_sa = use_sa
        bc = base_channels
        self.trajectory_head = TrajectoryHead(bc, 32)
        
        #encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, padding=1),
            nn.GroupNorm(1, bc, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc2 = BlurPoolDown(bc, bc * 2)
        self.enc3 = BlurPoolDown(bc * 2, bc * 4)
        self.enc4 = BlurPoolDown(bc * 4, bc * 8)

        #multi-scale branch
        self.downsample = nn.AvgPool2d(2)
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, padding=1),
            nn.GroupNorm(1, bc, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(bc, bc * 2, 3, padding=1),
            nn.GroupNorm(1, bc * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc_down3 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, 3, padding=1),
            nn.GroupNorm(1, bc * 4, affine=True),
            nn.ReLU(inplace=True),
        )
        self.match_channels = nn.Conv2d(bc * 4, bc * 8, kernel_size=1)
        self.fuse_norm = nn.GroupNorm(1, bc * 8, affine=True)

        self.bottleneck = nn.Sequential(
            ResidualBlock(bc * 8),
            ResidualBlock(bc * 8),
            ResidualBlock(bc * 8),
            ResidualBlock(bc * 8),
        )
        
        # Compressed skip to prevent identity copying
        self.skip_proj1 = nn.Sequential(
            nn.Conv2d(bc, bc // 2, 1),
            nn.GroupNorm(1, bc // 2),
            nn.ReLU(inplace=True),
        )
        
        #attention
        self.branch_channels = bc * 8
        if self.use_sa and args.use_edge_attention:
            self.edge_attention = EdgeAwareAttention(in_c=self.branch_channels, num_feat=self.num_feat)
            self.edge_proj = nn.Conv2d(self.num_feat, self.branch_channels, 1)
        self.w_edge = nn.Parameter(torch.tensor(args.w_edge, dtype=torch.float32))

        self.skip_gate3 = nn.Sequential(nn.Conv2d(bc * 4, bc * 4, 1), nn.Sigmoid())
        self.skip_gate2 = nn.Sequential(nn.Conv2d(bc * 2, bc * 2, 1), nn.Sigmoid())

        #decoder
        self.dec3 = UpsampleBlock(bc * 8, bc * 4, bc * 4)
        self.dec2 = UpsampleBlock(bc * 4, bc * 2, bc * 2)
        self.dec1 = UpsampleBlock(bc * 2, bc // 2, bc)
        self.out_conv = nn.Conv2d(bc, out_channels, 3, padding=1)

        #blur head
        self.blur_head = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
    
    def displacement(self, x, trajectory, max_disp=64.0):
        B, C, H, W = x.shape
        device = x.device
        
        dx_dy = trajectory[:, :2]
        confidence = trajectory[:, 2:3]
        
        offset_px = dx_dy * max_disp
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, H, W)
        
        offset_norm = torch.zeros_like(offset_px)
        offset_norm[:, 0] = offset_px[:, 0] / (W / 2.0)
        offset_norm[:, 1] = offset_px[:, 1] / (H / 2.0)
        
        offset_norm = offset_norm * confidence
        
        warped = F.grid_sample(
            x,
            (base_grid + offset_norm).permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )
        
        return warped

    def forward(self, x, stage=3):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        xd = self.downsample(x)
        d1 = self.enc_down1(xd)
        d2 = self.enc_down2(d1)
        d3 = self.enc_down3(d2)
        d3_down = F.avg_pool2d(d3, kernel_size=4, stride=4)
        fused = self.fuse_norm(e4 + self.match_channels(d3_down))

        fused = self.bottleneck(fused)
        attn_out = fused
        if self.use_sa and hasattr(self, "edge_attention"):
            edge_dict = self.edge_attention(fused)
            attn_out = attn_out + self.w_edge * self.edge_proj(edge_dict["out"])

        g3 = self.skip_gate3(e3)
        g2 = self.skip_gate2(e2)

        dec3_out = self.dec3(attn_out, e3 * g3)
        dec2_out = self.dec2(dec3_out, e2 * g2)
        compressed_skip = self.skip_proj1(e1)
        dec1_out = self.dec1(dec2_out, compressed_skip)

        trajectory_raw = self.trajectory_head(e1.detach())
        trajectory = torch.cat([
            torch.tanh(trajectory_raw[:, :2]),
            torch.sigmoid(trajectory_raw[:, 2:3])], dim=1)
        warped_input = self.displacement(x, trajectory)
        
        residual = self.out_conv(dec1_out)
        if self.out_activation == "tanh":
            residual = torch.tanh(residual)
            out = torch.clamp(warped_input + residual, -1.0, 1.0)
        else:
            residual = torch.sigmoid(residual)
            out = torch.clamp(warped_input + residual, 0.0, 1.0)
        
        blur_prob = torch.sigmoid(self.blur_head(residual))
        blur_prob = torch.clamp(blur_prob, 0.0, 1.0)

        modulation = torch.clamp(1.0 + 0.5 * (blur_prob - 0.5), 0.75, 1.25)
        if self.out_activation == "tanh":
            guided_out = torch.clamp(out * modulation, -1.0, 1.0)
        else:
            guided_out = torch.clamp(out * modulation, 0.0, 1.0)

        return {"out": guided_out, "blur_prob": blur_prob, "trajectory": trajectory, "warped": warped_input}