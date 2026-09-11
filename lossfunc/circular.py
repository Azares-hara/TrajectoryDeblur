import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sobel_edges

class CircularBlurRefinementLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, fake, blur_input, blur_prob=None):
        blur_edges = sobel_edges(blur_input)
        fake_edges = sobel_edges(fake)

        def grad_spatial(x):
            dx = x[..., :, 1:] - x[..., :, :-1]
            dy = x[..., 1:, :] - x[..., :-1, :]
            dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
            dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
            return dx, dy

        edx, edy = grad_spatial(blur_edges)
        edxx, _ = grad_spatial(edx)
        _, edyy = grad_spatial(edy)
        curvature = torch.abs(edxx) + torch.abs(edyy)

        #local mean/std (7×7)
        curv_mean = F.avg_pool2d(curvature, kernel_size=7, stride=1, padding=3)
        curv_std = torch.sqrt(
            F.avg_pool2d((curvature - curv_mean) ** 2, kernel_size=7, stride=1, padding=3) + 1e-6
        )
        circ_mask = torch.sigmoid((curvature - curv_mean) / (curv_std + 1e-6))

        if blur_prob is not None:
            circ_mask = circ_mask * torch.sigmoid(blur_prob)

        #edge deficit: fake is weaker than blur (under-deblurred)
        edge_deficit = F.relu(blur_edges - fake_edges)
        #edge excess: fake is much stronger than blur (ghost / duplicate edge)
        edge_excess = F.relu(fake_edges - blur_edges * 1.5)
        sharpen_loss = (edge_deficit * circ_mask).mean() + (edge_excess * circ_mask).mean()

        fdx, fdy = grad_spatial(fake_edges)
        bdx, bdy = grad_spatial(blur_edges)
        f_mag = torch.sqrt(fdx**2 + fdy**2 + 1e-8)
        b_mag = torch.sqrt(bdx**2 + bdy**2 + 1e-8)
        cos_dir = (fdx * bdx + fdy * bdy) / (f_mag * b_mag + 1e-8)
        geom_loss = ((1.0 - cos_dir) * circ_mask.squeeze(1)).mean()

        return self.weight * (sharpen_loss + 0.5 * geom_loss)