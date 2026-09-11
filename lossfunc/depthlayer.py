import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthLayerLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        lap = torch.tensor(
            [[0.,  1., 0.],
             [1., -4., 1.],
             [0.,  1., 0.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap)

        sx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)

    def _to_gray(self, img):
        return img.mean(dim=1, keepdim=True)

    def _gradient_mag(self, img):
        gray = self._to_gray(img)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def _laplacian(self, img):
        gray = self._to_gray(img)
        return F.conv2d(gray, self.lap_kernel, padding=1)

    def _local_freq_energy(self, img, window=16):
        gray = self._to_gray(img)
        coarse = F.avg_pool2d(gray, kernel_size=window, stride=1, padding=window // 2)
        coarse = coarse[:, :, :gray.shape[2], :gray.shape[3]]
        return torch.abs(gray - coarse)

    def estimate_depth_layer(self, blurry_input, blur_prob=None):
        grad_mag = self._gradient_mag(blurry_input)
        hf_energy = self._local_freq_energy(blurry_input)

        #Batch-wise normalization (stable across the mini-batch)
        def norm(x):
            mn = x.min()
            mx = x.max()
            return (x - mn) / (mx - mn + 1e-8)

        blur_from_grad = 1.0 - norm(grad_mag)
        blur_from_freq = 1.0 - norm(hf_energy)

        if blur_prob is not None:
            if blur_prob.shape[2:] != blurry_input.shape[2:]:
                blur_prob = F.interpolate(
                    blur_prob, size=blurry_input.shape[2:],
                    mode='bilinear', align_corners=False
                )
            depth_layer = (
                0.5 * blur_prob +
                0.3 * blur_from_grad +
                0.2 * blur_from_freq
            )
        else:
            depth_layer = 0.5 * blur_from_grad + 0.5 * blur_from_freq
        return depth_layer.detach()

    def forward(self, fake, real_input, blur_prob=None):
        depth_layer = self.estimate_depth_layer(real_input, blur_prob)

        input_grad = self._gradient_mag(real_input)
        output_grad = self._gradient_mag(fake)

        fg_mask = depth_layer
        bg_mask = 1.0 - depth_layer

        fg_sharpening = torch.mean(fg_mask * F.relu(input_grad - output_grad))
        bg_smoothness = torch.mean(bg_mask * F.relu(output_grad - input_grad * 1.2))

        depth_grad_x = torch.abs(
            depth_layer[:, :, :, 1:] - depth_layer[:, :, :, :-1]
        )
        depth_grad_y = torch.abs(
            depth_layer[:, :, 1:, :] - depth_layer[:, :, :-1, :]
        )
        depth_grad_x = F.pad(depth_grad_x, (0, 1))
        depth_grad_y = F.pad(depth_grad_y, (0, 0, 0, 1))
        boundary_mask = torch.sqrt(
            depth_grad_x ** 2 + depth_grad_y ** 2 + 1e-6
        ).detach()

        output_lap = self._laplacian(fake)
        boundary_loss = torch.mean(
            boundary_mask * torch.abs(output_lap)
        )

        expected_sharpness = (
            fg_mask * output_grad.mean(dim=(1,2,3), keepdim=True) * 1.5 +
            bg_mask * output_grad.mean(dim=(1,2,3), keepdim=True) * 0.7
        ).detach()
        consistency_loss = F.l1_loss(output_grad, expected_sharpness)

        total = fg_sharpening + 0.5 * bg_smoothness + 0.3 * boundary_loss + 0.2 * consistency_loss
        return self.weight * total, {
            "fg_sharp": fg_sharpening.item(),
            "bg_smooth": bg_smoothness.item(),
            "boundary": boundary_loss.item(),
            "consistency": consistency_loss.item(),
        }