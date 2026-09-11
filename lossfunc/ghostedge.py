import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GhostEdgeLoss(nn.Module):
    def __init__(self, weight=0.15, proximity_px=8, angle_threshold_deg=20.0):
        super().__init__()
        self.weight = weight
        self.proximity_px = proximity_px
        self.angle_threshold_rad = math.radians(angle_threshold_deg)

        sx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=torch.float32)
        sy = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)

    def _gradients(self, img):
        gray = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return gx, gy, mag

    def _estimate_blur_vector(self, real_input):
        gx, gy, mag = self._gradients(real_input)
        weight = mag.detach()
        mean_gx = (gx * weight).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) + 1e-8)
        mean_gy = (gy * weight).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) + 1e-8)
        blur_angle = torch.atan2(mean_gy, mean_gx) + math.pi / 2
        return blur_angle

    def _orientation_ghost_penalty(self, fake, real_input):
        gx_out, gy_out, mag_out = self._gradients(fake)
        gx_in, gy_in, mag_in = self._gradients(real_input)

        orientation = torch.atan2(gy_out, gx_out)
        blur_angle = self._estimate_blur_vector(real_input).view(-1, 1, 1, 1)

        angle_to_blur = torch.abs(orientation - blur_angle)
        angle_to_blur = torch.min(angle_to_blur,
                                  torch.tensor(math.pi, device=fake.device) - angle_to_blur)
        aligned_with_blur = (angle_to_blur < self.angle_threshold_rad).float().detach()

        p = self.proximity_px
        #horizontal proximity
        shifted_h = F.pad(orientation[:, :, :, p:], (0, p, 0, 0))
        angle_diff_h = torch.abs(orientation - shifted_h)
        angle_diff_h = torch.min(angle_diff_h,
                                 torch.tensor(math.pi, device=fake.device) - angle_diff_h)
        same_orientation_h = (angle_diff_h < self.angle_threshold_rad).float().detach()

        #vertical proximity
        shifted_v = F.pad(orientation[:, :, p:, :], (0, 0, 0, p))
        angle_diff_v = torch.abs(orientation - shifted_v)
        angle_diff_v = torch.min(angle_diff_v,
                                 torch.tensor(math.pi, device=fake.device) - angle_diff_v)
        same_orientation_v = (angle_diff_v < self.angle_threshold_rad).float().detach()

        #weight by blur direction (horizontal blur → weight horizontal check more)
        blur_cos = torch.abs(torch.cos(blur_angle))
        blur_sin = torch.abs(torch.sin(blur_angle))
        same_orientation = blur_cos * same_orientation_h + blur_sin * same_orientation_v

        blur_mask = (mag_in > mag_in.mean()).float().detach()
        binary_edges = (mag_out > mag_out.mean()).float().detach()

        ghost_penalty = torch.mean(
            blur_mask * aligned_with_blur * same_orientation * binary_edges * mag_out
        )
        return ghost_penalty

    def _scale_space_penalty(self, fake, real_input):
        gx_in, gy_in, mag_in = self._gradients(real_input)
        blur_mask = (mag_in > mag_in.mean()).float().detach()

        _, _, mag_full = self._gradients(fake)
        fake_half = F.avg_pool2d(fake, kernel_size=2, stride=2)
        _, _, mag_half = self._gradients(fake_half)
        mag_half_up = F.interpolate(
            mag_half, size=mag_full.shape[2:], mode='bilinear', align_corners=False
        )

        ghost_candidates = F.relu(mag_full - mag_half_up * 1.2)  
        return torch.mean(blur_mask * ghost_candidates)

    def forward(self, fake, real_input):
        orientation_loss = self._orientation_ghost_penalty(fake, real_input)
        scale_space_loss = self._scale_space_penalty(fake, real_input)
        total = orientation_loss + 0.5 * scale_space_loss
        return self.weight * total