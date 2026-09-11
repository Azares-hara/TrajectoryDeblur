import torch
import torch.nn as nn
import torch.nn.functional as F
from model2.SelfAttention import EdgeAwareAttention


class BlurPoolDown(nn.Module):
    """
    Learnable blur kernel with strong Gaussian initialization.
    Adapts downsampling to preserve small-object details while anti-aliasing.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.blur = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=1,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        with torch.no_grad():
            k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
            k = k / k.sum()
            self.blur.weight.copy_(k.expand(in_channels, 1, 3, 3))
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(1, base_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.traj_gate = nn.Conv2d(base_channels, 1, 1)
        with torch.no_grad():
            self.traj_gate.bias.fill_(0.0)

    def forward(self, x, stage=3):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        gate = torch.sigmoid(self.traj_gate(d2))
        return self.out(d2 + gate * e1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels, affine=True, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels, affine=True, eps=1e-5)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out * self.se(out)
        return F.relu(out + residual)


class TraUpsampleBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip_gate = nn.Conv2d(skip_ch, skip_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        gate = torch.sigmoid(self.skip_gate(skip)) * 0.5 + 0.5
        x = torch.cat([x, skip * gate], dim=1)
        return self.conv(x)


class TraUNetGenerator(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3,
                 base_channels=64, use_sa=False, out_activation="tanh"):
        super().__init__()
        self.num_feat = getattr(args, 'num_feat', 64)
        self.out_activation = out_activation
        self.use_sa = use_sa
        bc = base_channels

        # 20 dilated ResBlocks (fixed architecture)
        blocks = []
        for i in range(20):
            if i < 6:
                blocks.append(ResidualBlock(bc * 8, dilation=1))
            elif i < 13:
                blocks.append(ResidualBlock(bc * 8, dilation=2))
            elif i < 17:
                blocks.append(ResidualBlock(bc * 8, dilation=2))
            else:
                blocks.append(ResidualBlock(bc * 8, dilation=1))
        self.bottleneck = nn.Sequential(*blocks)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, padding=1),
            nn.GroupNorm(1, bc, affine=True),
            nn.ReLU(inplace=True),
        )
        self.warped_proj = nn.Conv2d(3, bc, kernel_size=1, bias=False)
        nn.init.zeros_(self.warped_proj.weight)
        self.enc2 = BlurPoolDown(bc, bc * 2)
        self.enc3 = BlurPoolDown(bc * 2, bc * 4)
        self.enc4 = BlurPoolDown(bc * 4, bc * 8)

        # Trajectory head (fine, from e1)
        self.trajectory_head = TrajectoryHead(in_channels=bc, base_channels=32)
        self.kernel_embed = nn.Conv2d(3, bc, 1)

        # NEW: Global motion correction from bottleneck (e4)
        self.global_motion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bc * 8, bc * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bc * 2, 3, 1),
        )
        with torch.no_grad():
            self.global_motion[-1].bias.fill_(0.0)

        # Decoder
        self.dec3 = TraUpsampleBlock(bc * 8, bc * 4, bc * 4)
        self.dec2 = TraUpsampleBlock(bc * 4, bc * 2, bc * 2)
        self.dec1 = TraUpsampleBlock(bc * 2, bc, bc)

        self.out_conv = nn.Conv2d(bc, out_channels, 3, padding=1)
        self.residual_gain = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

        # Edge-aware attention
        self.branch_channels = bc * 8
        if self.use_sa and getattr(args, 'use_edge_attention', False):
            self.edge_attention = EdgeAwareAttention(
                in_c=self.branch_channels, num_feat=self.num_feat
            )
            self.edge_proj = nn.Conv2d(self.num_feat, self.branch_channels, 1)
            self.w_edge = nn.Parameter(
                torch.tensor(getattr(args, 'w_edge', 0.1), dtype=torch.float32)
            )

        # Blur & edge heads
        self.blur_head = nn.Sequential(
            nn.Conv2d(bc, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.edge_head_conv1 = nn.Sequential(
            nn.Conv2d(bc, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
        )
        self.edge_head_out = nn.Conv2d(32, 1, kernel_size=1)
        with torch.no_grad():
            self.edge_head_out.bias.fill_(0.0)
        self.edge_head_skip = nn.Conv2d(bc, 1, kernel_size=1)
        with torch.no_grad():
            self.edge_head_skip.weight.fill_(0.0)
            self.edge_head_skip.bias.fill_(0.0)

    def displacement(self, x, trajectory, max_disp_ratio=0.33):
        B, C, H, W = x.shape
        device = x.device
        dx_dy = trajectory[:, :2]
        confidence = trajectory[:, 2:3]

        max_disp = min(W, H) * max_disp_ratio
        offset_px = dx_dy * max_disp

        max_safe_px = min(W, H) * 0.4
        offset_px = torch.clamp(offset_px, -max_safe_px, max_safe_px)

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
        # === PASS 1: Initial trajectory from shallow features ===
        e1_raw = self.enc1(x)
        trajectory_raw = self.trajectory_head(e1_raw, stage=stage)
        trajectory = torch.cat([
            torch.tanh(trajectory_raw[:, :2]),      # dx, dy in [-1, 1]
            torch.sigmoid(trajectory_raw[:, 2:3])   # confidence
        ], dim=1)

        # Initial warp
        warped_input = self.displacement(x, trajectory)

        # Encoder with initial warp
        e1 = e1_raw + self.warped_proj(warped_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # === PASS 2: Global motion correction from bottleneck ===
        B, _, H, W = trajectory.shape
        global_traj = self.global_motion(e4)          # B, 3, 1, 1
        global_traj = global_traj.expand(B, 3, H, W)

        # Residual refinement: small global correction added to local trajectory
        trajectory_refined = trajectory + 0.1 * torch.cat([
            torch.tanh(global_traj[:, :2]),
            torch.sigmoid(global_traj[:, 2:3])
        ], dim=1)

        # Re-warp with refined trajectory for skip connection and final output
        warped_input = self.displacement(x, trajectory_refined)
        trajectory = trajectory_refined               # Use refined downstream

        # Edge-aware attention on bottleneck
        b_in = e4
        use_edge = self.use_sa and getattr(self, 'edge_attention', None) is not None
        if use_edge:
            edge_out = self.edge_attention(b_in)
            b_in = b_in + self.w_edge * self.edge_proj(edge_out["out"])

        # Bottleneck & decoder
        b = self.bottleneck(b_in)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Final modulation with refined trajectory
        kernel_feat = self.kernel_embed(trajectory)
        d1 = d1 * (1 + 0.25 * kernel_feat)

        # High-resolution edge map
        edge_feat = self.edge_head_conv1(d1)
        edge_map = torch.sigmoid(self.edge_head_out(edge_feat) + self.edge_head_skip(d1))
        blur_prob = self.blur_head(d1)

        # Residual output
        residual = self.out_conv(d1) * self.residual_gain
        out = torch.clamp(warped_input + residual, -1.0, 1.0)

        return {
            "out": out,
            "trajectory": trajectory,
            "warped": warped_input,
            "blur_prob": blur_prob,
            "edge_map": edge_map,
        }
