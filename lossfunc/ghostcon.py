%%writefile /kaggle/working/Deblur/lossfunc/ghostcon.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalGhostConsensus(nn.Module):
    def __init__(self, max_shift=16, num_samples=8):
        super().__init__()
        self.max_shift = max_shift
        self.num_samples = num_samples
        
    def forward(self, blur_input, blur_direction, blur_extent):
        B, C, H, W = blur_input.shape
        t_vals = torch.linspace(-1, 1, self.num_samples, device=blur_input.device)
        
        consensus_stack = []
        weight_stack = []
        
        for t in t_vals:
            shift_pixels = t * blur_extent * self.max_shift
            
            dx = blur_direction[:, 0:1] * shift_pixels
            dy = blur_direction[:, 1:2] * shift_pixels
            
            y_base, x_base = torch.meshgrid(
                torch.linspace(-1, 1, H, device=blur_input.device),
                torch.linspace(-1, 1, W, device=blur_input.device),
                indexing='ij'
            )
            grid_y = y_base + 2.0 * dy.squeeze(1) / H
            grid_x = x_base + 2.0 * dx.squeeze(1) / W
            grid = torch.stack([grid_x, grid_y], dim=-1)
            
            warped = F.grid_sample(
                blur_input, grid, 
                mode='bilinear', padding_mode='reflection', align_corners=False
            )
            
            center = blur_input
            agreement = torch.exp(-10.0 * F.l1_loss(warped, center, reduction='none').mean(dim=1, keepdim=True))
            
            consensus_stack.append(warped)
            weight_stack.append(agreement)
        
        consensus_stack = torch.stack(consensus_stack, dim=0)
        weight_stack = torch.stack(weight_stack, dim=0)
        
        weights = weight_stack / (weight_stack.sum(dim=0, keepdim=True) + 1e-6)
        consensus_sharp = (consensus_stack * weights).sum(dim=0)
        confidence = weight_stack.mean(dim=0)
        
        return consensus_sharp, confidence


class DirectionalBlurPriorLoss(nn.Module):
    def __init__(self, weight=1.0, max_shift=16):
        super().__init__()
        self.weight = weight
        self.consensus_module = DirectionalGhostConsensus(max_shift=max_shift)
        
    def forward(self, deblurred, blur_input, blur_direction, blur_extent, blur_prob, valid_mask=None):
        consensus_sharp, confidence = self.consensus_module(blur_input, blur_direction, blur_extent)
    
        mask = blur_prob * confidence
        if valid_mask is not None:
            mask = mask * valid_mask
        mask = mask.clamp(0, 1)
        
        #Spatial domain loss (mask that matches spatial dimensions)
        diff = F.l1_loss(deblurred, consensus_sharp, reduction='none')
        weighted_diff = (diff.mean(dim=1, keepdim=True) * mask).sum() / (mask.sum() + 1e-6)
        
        #Frequency domain loss — apply mask by resizing to FFT dimensions
        #/compute frequency loss on full images without masking 
        fft_deblurred = torch.fft.rfft2(deblurred.float(), norm="ortho")
        fft_consensus = torch.fft.rfft2(consensus_sharp.float(), norm="ortho")
        freq_diff = F.l1_loss(
            torch.abs(fft_deblurred), 
            torch.abs(fft_consensus)
        ) 
        
        #H_fft, W_fft = fft_deblurred.shape[-2:]
        #mask_fft = F.interpolate(mask, size=(H_fft, W_fft), mode='nearest')
        #freq_diff = F.l1_loss(torch.abs(fft_deblurred), torch.abs(fft_consensus), reduction='none')
        #freq_loss = (freq_diff.mean(dim=1, keepdim=True) * mask_fft).sum() / (mask_fft.sum() + 1e-6)
        
        total = weighted_diff + 0.1 * freq_diff
        return self.weight * total, {
            'pixel_loss': weighted_diff.item(),
            'freq_loss': freq_diff.item(),
            'mean_confidence': confidence.mean().item(),
            'mean_mask': mask.mean().item()
        }
