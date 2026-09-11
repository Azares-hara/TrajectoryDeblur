import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurDirectionLoss(nn.Module):
    def __init__(self, weight=0.1, window=5):
        super().__init__()
        self.weight = weight
        self.window = window
        
        kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
        ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        
    def forward(self, blur_img, pred_motion, blur_prob):
        B, C, H, W = blur_img.shape
        device = blur_img.device
        
        gray = blur_img.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kx.to(device), padding=1)
        gy = F.conv2d(gray, self.ky.to(device), padding=1)
        
        Jxx = gx * gx
        Jxy = gx * gy
        Jyy = gy * gy
        
        pad = self.window // 2
        Jxx = F.avg_pool2d(Jxx, self.window, stride=1, padding=pad)
        Jxy = F.avg_pool2d(Jxy, self.window, stride=1, padding=pad)
        Jyy = F.avg_pool2d(Jyy, self.window, stride=1, padding=pad)
        
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        tmp = torch.sqrt(trace**2 - 4*det + 1e-6)
        lambda_min = 0.5 * (trace - tmp)
        v_x = Jxy
        v_y = lambda_min - Jxx
        norm = torch.sqrt(v_x**2 + v_y**2 + 1e-6)
        
        with torch.no_grad():
            target_motion = torch.cat([v_x / norm, v_y / norm], dim=1)
            edge_mag = torch.sqrt(Jxx + Jyy + 1e-6)
            edge_mask = (edge_mag > edge_mag.mean()).float()
        
        dot = (pred_motion * target_motion).sum(dim=1, keepdim=True)
        mask = blur_prob * edge_mask
        loss = ((1.0 - dot) * mask).sum() / (mask.sum() + 1e-6)
        return loss * self.weight