import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as lpips_lib


class LPIPS(nn.Module):
    def __init__(self, net='squeeze', device='cuda:0', use_half=False):
        super().__init__()
        self.device   = device
        self.net_name = net
        self.use_half = use_half
        self.lpips_fn = lpips_lib.LPIPS(net=net).to(device)
        if use_half:
            self.lpips_fn = self.lpips_fn.half()

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        if next(self.lpips_fn.parameters()).dtype == torch.float16:
            x, y = x.half(), y.half()
        else:
            x, y = x.float(), y.float()

        if x.ndim == 3:
            x = x.unsqueeze(0)
        if y.ndim == 3:
            y = y.unsqueeze(0)

        val = self.lpips_fn(x, y)
        val = torch.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)
        return val.mean()


class GradientLoss(nn.Module):
    """
    Penalises differences in image gradients between output and target.
    """

    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device

        kernel_x = torch.tensor([[-1.,  0.,  1.],[-2.,  0.,  2.],[-1.,  0.,  1.]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1., -2., -1.],[ 0.,  0.,  0.],[ 1.,  2.,  1.]], dtype=torch.float32)

        self.register_buffer("weight_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("weight_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, output, target):
        device_type = output.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            output_gray = output.mean(dim=1, keepdim=True).float()
            target_gray = target.mean(dim=1, keepdim=True).float()

            wx = self.weight_x.to(output_gray.device).float()
            wy = self.weight_y.to(output_gray.device).float()

            grad_out_x = F.conv2d(output_gray, wx, padding=1)
            grad_out_y = F.conv2d(output_gray, wy, padding=1)
            grad_tar_x = F.conv2d(target_gray, wx, padding=1)
            grad_tar_y = F.conv2d(target_gray, wy, padding=1)

            loss_x = F.l1_loss(grad_out_x, grad_tar_x)
            loss_y = F.l1_loss(grad_out_y, grad_tar_y)
            return loss_x + loss_y
