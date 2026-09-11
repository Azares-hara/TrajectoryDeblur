import torch
import torch.nn as nn
import torch.nn.functional as F

class SharpnessPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.,  0.,  1.],
             [-2.,  0.,  2.],
             [-1.,  0.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _gradients(self, x):
        C = x.size(1)
        kx = self.kernel_x.repeat(C, 1, 1, 1)
        ky = self.kernel_y.repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, blur_prob, fake_real, input_real):
        g_fake = self._gradients(fake_real)
        g_input = self._gradients(input_real)
        sharp_gain = g_fake - g_input

        if blur_prob.shape[2:] != sharp_gain.shape[2:]:
            sharp_gain = F.interpolate(
                sharp_gain, size=blur_prob.shape[2:],
                mode='bilinear', align_corners=False
            )

        #under-sharp: output edge weaker than input (failed deblur)
        under_sharp = F.relu(-sharp_gain)
        #overver-sharp / ghost: output edge much stronger than input (duplicate edge)
        
        over_sharp = F.relu(sharp_gain - 0.6)

        return torch.mean(blur_prob * (under_sharp + 0.5 * over_sharp))

_criterion = None

def sharpness_prior_loss(blur_prob, fake_real, input_real):
    global _criterion
    if _criterion is None or next(_criterion.parameters(), None) is None:
        _criterion = SharpnessPriorLoss().to(fake_real.device)
    if next(iter(_criterion.buffers())).device != fake_real.device:
        _criterion = _criterion.to(fake_real.device)
    return _criterion(blur_prob, fake_real, input_real)