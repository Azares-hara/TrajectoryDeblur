import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurBandLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        kernel = torch.tensor(
            [[0.,  1., 0.],
             [1., -4., 1.],
             [0.,  1., 0.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', kernel)

    def _transition_width(self, img):
        gray = img.mean(dim=1, keepdim=True)
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def _laplacian(self, img):
        gray = img.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.lap_kernel, padding=1)

    def forward(self, fake, real_input):
        input_grad = self._transition_width(real_input)
        output_grad = self._transition_width(fake)
        #LOCAL adaptive threshold: blur where local input gradient is above local mean
        local_mean = F.avg_pool2d(input_grad, kernel_size=15, stride=1, padding=7)
        blur_mask = (input_grad > local_mean * 0.8).float().detach()
        output_lap = self._laplacian(fake)
        #border loss:
        border_loss = torch.mean(blur_mask * torch.abs(output_lap))

        #sharpness loss:
        sharpness_loss = torch.mean(blur_mask * F.relu(input_grad - output_grad))

        return self.weight * (border_loss + sharpness_loss)