%%writefile /kaggle/working/Deblur/train.py
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataset.common
from dataset.gopro_large import GOPRO_Large, SharpDataset
from utils import (
    MultiSaver, get_disc_weights, match_size,
    apply_random_blur, sobel_edges, save_tensor_as_image,
)
from lossfunc.frequencyloss import FrequencyLoss
from lossfunc.metric import PSNR, SSIM, LPIPSMetric, NIQEMetric, BRISQUEMetric
from lossfunc import get_default_device, NeighborLoss
from lossfunc.adversarial import Adversarial
from lossfunc.lpips import GradientLoss, LPIPS
from lossfunc.lpipsregistry import get_lpips
from lossfunc.blurband import BlurBandLoss
from lossfunc.depthlayer import DepthLayerLoss
from lossfunc.ghostedge import GhostEdgeLoss
from model2.discriminator import MultiScaleDiscriminator
from torchvision.utils import save_image
from model2.SelfAttention import blur_loss as compute_blur_loss
from lossfunc.metric import safe_fft_loss
from lossfunc.sharpprior import sharpness_prior_loss
from lossfunc.circular import CircularBlurRefinementLoss

from utils import save_trajectory_visualization

def line_consistency_loss(fake, ref):
    return F.l1_loss(sobel_edges(fake), sobel_edges(ref))


def centroid_alignment_loss(fake, ref):
    fake_edges = sobel_edges(fake)
    ref_edges  = sobel_edges(ref)

    def weighted_centroid(edges):
        B, _, H, W = edges.shape
        y_coords = torch.arange(H, device=edges.device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=edges.device).view(1, 1, W).expand(B, H, W)
        weights  = edges.squeeze(1) + 1e-6
        cx = (x_coords * weights).sum(dim=(1,2)) / weights.sum(dim=(1, 2))
        cy = (y_coords * weights).sum(dim=(1,2)) / weights.sum(dim=(1, 2))
        return torch.stack([cx, cy], dim=1)

    return F.mse_loss(weighted_centroid(fake_edges), weighted_centroid(ref_edges))


def fft_loss(pred, target):
    pred_fft   = torch.fft.rfft2(pred.float(),norm="ortho")
    target_fft = torch.fft.rfft2(target.float(),norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class Trainer:
    def __init__(self, args, model, criterion, optimizer):
        print('===> Initializing trainer')
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        
        blur_dataset = GOPRO_Large(args, mode='train')
        sharp_dataset = SharpDataset(args, mode='train')
        val_dataset = GOPRO_Large(args, mode='val')
        test_dataset = GOPRO_Large(args, mode='test')

        num_workers = getattr(args, 'num_workers', 2)
        pin_memory = str(args.device).startswith('cuda')

        self.loaders = {
            'train_blur':DataLoader(blur_dataset,  batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=2 if num_workers > 0 else None),
            'train_sharp': DataLoader(sharp_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=2 if num_workers > 0 else None),
            'val':         DataLoader(val_dataset,   batch_size=args.val_batch_size,
                              shuffle=False, num_workers=1,
                              pin_memory=pin_memory,
                              persistent_workers=True,
                              prefetch_factor=1),
            'test':        DataLoader(test_dataset,  batch_size=args.val_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=2 if num_workers > 0 else None),
        }
        self.real_loader = self.loaders['train_blur']
        self.sharp_iter  = iter(self.loaders['train_sharp'])

        self.tb_logger = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))
        self.model = model.to(torch.device(args.device))
        self.adversarial = Adversarial(args, model, optimizer)
        self.epoch = args.start_epoch
        self.save_dir = args.save_dir
        self.device = torch.device(args.device)
        self.dtype = args.dtype
        self.dtype_eval = torch.float32

        if hasattr(self.model, '_inner'):
            disc = self.model._inner('D')
            if disc is not None:
                self.discriminator = disc.to(self.device)
            else:
                self.discriminator = MultiScaleDiscriminator(n_feats=args.n_feats).to(self.device)
        else:
            #fallback for raw generator 
            if hasattr(self.model, 'model') and 'D' in self.model.model and self.model.model['D'] is not None:
                self.discriminator = self.model.model['D']
            else:
                self.discriminator = MultiScaleDiscriminator(n_feats=args.n_feats).to(self.device)

        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=args.lr_D, betas=(0.5, 0.999))
        self.freq_loss_module = FrequencyLoss(
            weight=args.lambda_freq,
            use_phase=True,
            multi_scale=True,
            scales=(2, 4),
            return_components=False,
        )
        self.blur_band_loss   = BlurBandLoss(weight=0.05).to(self.device)
        self.depth_layer_loss = DepthLayerLoss(weight=0.05).to(self.device)
        self.ghost_edge_loss  = GhostEdgeLoss(weight=0.02, proximity_px=12).to(self.device)
        self.circular_loss = CircularBlurRefinementLoss(weight=1.0).to(self.device)
        
        self.neighbor_loss = NeighborLoss()
        self.psnr_metric = PSNR(device=self.device)
        self.ssim_metric = SSIM(device_type=self.device.type)
        self.niqe_metric = NIQEMetric()
        self.brisque_metric = BRISQUEMetric()
        self.grad_loss = GradientLoss(device=self.device).to(torch.float32)
        self.lpips = get_lpips(net="squeeze", device=self.device, use_half=True)
        self.lpips_metric = self.safe_lpips
 
        self.result_dir = (
            args.demo_output_dir if args.demo and args.demo_output_dir
            else os.path.join(args.save_dir, 'result')
        )
        os.makedirs(self.result_dir, exist_ok=True)
        print(f'Results are saved in {self.result_dir}')

        self.imsaver = MultiSaver(self.result_dir)
        self.scaler  = torch.amp.GradScaler(
            device='cuda', init_scale=self.args.init_scale, enabled=self.args.amp
        )

        if self.args.resume:
            self._resume()

        log_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./"
        self.log_file = open(os.path.join(log_dir, "train_log.txt"), "w")

    def _resume(self):
        model_dir = os.path.join(self.args.resume_dir, 'models')
        optim_dir = os.path.join(self.args.resume_dir, 'optim')

        if self.args.manual_load_epoch is not None and self.args.manual_load_epoch > 0:
            epoch = self.args.manual_load_epoch
            ckpt_model = os.path.join(model_dir, f"model-{epoch}.pt")
            ckpt_optim = os.path.join(optim_dir, f"optim-{epoch}.pt")
            if os.path.exists(ckpt_model) and os.path.exists(ckpt_optim):
                print(f"[INFO] Forcing manual load at epoch {epoch}")
                self.load(epoch=epoch)
                self.epoch = epoch + 1
                for opt_name in ["G", "D"]:
                    opt_obj = getattr(self.optimizer, opt_name, None)
                    if opt_obj is not None and hasattr(opt_obj, "scheduler"):
                        opt_obj.scheduler.step(epoch)
                return
            print(f"[WARNING] Manual epoch {epoch} not found. Falling back to auto-resume.")

        if not os.path.exists(model_dir):
            print("[WARNING] No models directory found. Starting from scratch.")
            self.epoch = 1
            return

        model_list = [n for n in os.listdir(model_dir) if n.startswith("model-")]
        last_epoch = 0
        for name in model_list:
            nums = re.findall(r"\d+", name)
            if nums:
                last_epoch = max(last_epoch, int(nums[0]))

        if last_epoch > 0:
            print(f"[INFO] Auto-resuming from epoch {last_epoch}")
            self.load(epoch=last_epoch)
            self.epoch = last_epoch + 1
            for opt_name in ["G", "D"]:
                opt_obj = getattr(self.optimizer, opt_name, None)
                if opt_obj is not None and hasattr(opt_obj, "scheduler"):
                    opt_obj.scheduler.step(last_epoch)
        else:
            print("[WARNING] No valid checkpoint found. Starting from scratch.")
            self.epoch = 1

    def save(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if epoch % self.args.save_every == 0:
            self.model.save(epoch)
            self.optimizer.save(epoch)

    def load(self, epoch=None, pretrained=None):
        if epoch is None:
            epoch = self.args.load_epoch
        self.model.load(epoch, pretrained)
        self.optimizer.load(epoch)

    def safe_lpips(self, img1, img2):
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        return torch.nan_to_num(self.lpips(img1, img2), nan=0.0).mean()

    def _next_sharp(self):
        try:
            batch = next(self.sharp_iter)
        except StopIteration:
            self.sharp_iter = iter(self.loaders['train_sharp'])
            batch = next(self.sharp_iter)
        return batch["sharp"].to(self.device, dtype=self.dtype_eval)  

    def train(self, start_epoch, num_epochs):
        self.model.train()
        stage = 3

        W_BLUR_PRIOR  = getattr(self.args, "lambda_blur",  0.05)
        W_PRIOR = getattr(self.args, "lambda_prior", 0.05)
        W_BLUR_BAND = 0.01
        W_DEPTH_LAYER = 0.01
        W_GHOST = 0.02
        W_GRAD = 0.1
        W_FFT = 0.5
        W_LPIPS = 0.0
        W_EDGE_CONS = 0.0
        W_CENTROID = 0.0
        W_L1 = 1.0

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            generator = self.model._inner('G') if hasattr(self.model, '_inner') else self.model            
            #phase 1: trajectory on
            if epoch < 60:
                for param in generator.parameters():
                    param.requires_grad = False
                for param in generator.trajectory_head.parameters():
                    param.requires_grad = True
            #phase 2: decoder + trajectory frozen 
            elif epoch < 75:
                for param in generator.parameters():
                    param.requires_grad = False
                for param in generator.trajectory_head.parameters():
                    param.requires_grad = False
                for param in generator.dec3.parameters():
                    param.requires_grad = True
                for param in generator.dec2.parameters():
                    param.requires_grad = True
                for param in generator.dec1.parameters():
                    param.requires_grad = True
                for param in generator.out_conv.parameters():
                    param.requires_grad = True
            #phase 3: full network, trajectory no
            else:
                for param in generator.parameters():
                    param.requires_grad = True
                for param in generator.trajectory_head.parameters():
                    param.requires_grad = False
            
            with tqdm(total=len(self.loaders['train_blur']), ncols=100,
                      desc=f"Epoch {self.epoch}") as tq:

                for idx, batch in enumerate(self.real_loader):
                    global_step = epoch * len(self.loaders['train_blur']) + idx

                    real_blur = batch["blur"]
                    real_sharp = batch["sharp"]

                    input_real, _ = dataset.common.to(
                        real_blur, None, device=self.device, dtype=self.dtype_eval
                    )
                    outputs = self.model(input_real, stage)
                    fake_real = outputs["out"]
                    
                    fake_real = match_size(fake_real, input_real)
                    trajectory = outputs.get("trajectory", None)
                    warped = outputs.get("warped", None)
                    blur_prob  = outputs.get("blur_prob", None)
                    sharp_real = (
                        real_sharp.to(self.device, dtype=self.dtype_eval)
                        if real_sharp is not None else None
                    )

                    fake_for_loss = torch.clamp(fake_real,  -1.0, 1.0)
                    input_for_loss = torch.clamp(input_real, -1.0, 1.0)
                    sharp_for_loss = (
                        torch.clamp(sharp_real, -1.0, 1.0) if sharp_real is not None else None
                    )
                    
                    if torch.isnan(fake_real).any():
                        print(f"[Epoch {epoch}, Batch {idx}] NaN detected -- skipping.")
                        continue
                    
                    #frequency gain  
                    fft_fake = torch.fft.rfft2(fake_for_loss.float(),  norm="ortho")
                    fft_input = torch.fft.rfft2(input_for_loss.float(), norm="ortho")
                    freq_gain = (
                        torch.mean(torch.abs(fft_fake)) -
                        torch.mean(torch.abs(fft_input))
                    )
                    freq_gain_loss = -0.01 * torch.clamp(freq_gain, min=-10.0, max=10.0)

                    #blur head losses
                    blur_loss_val = (
                        compute_blur_loss(blur_prob, fake_for_loss)
                        if blur_prob is not None
                        else torch.tensor(0.0, device=self.device)
                    )
                    prior_loss = (
                        sharpness_prior_loss(blur_prob, fake_real, input_real)
                        if blur_prob is not None
                        else torch.tensor(0.0, device=self.device)
                    )
                    
                    #blur band
                    blur_band_val = torch.tensor(0.0, device=self.device)
                    if epoch >= 10:
                        blur_band_val = self.blur_band_loss(fake_for_loss, input_for_loss)

                    #depth layer
                    depth_layer_val = torch.tensor(0.0, device=self.device)
                    depth_layer_info = {}
                    if epoch >= 20:
                        depth_layer_val, depth_layer_info = self.depth_layer_loss(
                            fake_for_loss, input_for_loss, blur_prob=blur_prob)

                    #ghost edge
                    ghost_edge_val = torch.tensor(0.0, device=self.device)
                    if epoch >= 150:
                        ghost_edge_val = self.ghost_edge_loss(fake_for_loss, input_for_loss)

                    #edge supervision
                    edge_map = outputs.get("edge_map", None)
                    edge_loss = torch.tensor(0.0, device=self.device)
                    if edge_map is not None and epoch >= 130:
                        edge_loss = torch.nan_to_num(
                            F.l1_loss(edge_map, sobel_edges(fake_for_loss)),
                            nan=0.0, posinf=1.0, neginf=-1.0,
                        )  
                    #circular
                    circ_loss = torch.tensor(0.0, device=self.device)
                    circ_weight = 0.0
                    if epoch >= 150: 
                        circ_weight = max(0.0, min(0.02, 0.005 + (epoch - 150) * 0.00075))
                        circ_loss = self.circular_loss(fake_for_loss, input_for_loss, blur_prob)

                    #trajectory losses
                    traj_smooth_loss = torch.tensor(0.0, device=self.device)
                    traj_align_loss = torch.tensor(0.0, device=self.device)
                    traj_magnitude_loss = torch.tensor(0.0, device=self.device)
                    dx_dy = None
                    
                    if trajectory is not None:
                        dx_dy = trajectory[:, :2]                      
                        if epoch < 100:  
                            conf = trajectory[:, 2:3]                   
                            dx_h = dx_dy[:, :, :, 1:] - dx_dy[:, :, :, :-1]
                            dx_v = dx_dy[:, :, 1:, :] - dx_dy[:, :, :-1, :]
                            traj_smooth_loss = (dx_h.abs().mean() + dx_v.abs().mean()) * 0.01
                            
                            if blur_prob is not None:
                                motion_weight = blur_prob
                                motion_mask = (blur_prob > 0.5).float()
                                traj_align_loss = F.mse_loss(conf * motion_mask, blur_prob * motion_mask) * 0.05
                                #reduced magnitude penalty
                                traj_magnitude_loss = (dx_dy.abs() * motion_weight).mean() * 0.005
                            else:
                                traj_magnitude_loss = dx_dy.abs().mean() * 0.005
                    
                    # Supervised losses
                    l1_loss = torch.tensor(0.0, device=self.device)
                    lpips_loss = torch.tensor(0.0, device=self.device)
                    grad_loss = torch.tensor(0.0, device=self.device)
                    fft_sup_loss = torch.tensor(0.0, device=self.device)
                    edge_consistency = torch.tensor(0.0, device=self.device)
                    centroid_loss = torch.tensor(0.0, device=self.device)
                    warp_loss = torch.tensor(0.0, device=self.device)
                    warp_psnr = 0.0
                    
                    if sharp_for_loss is not None:
                        l1_loss = F.l1_loss(fake_for_loss, sharp_for_loss)
                        lpips_loss = torch.nan_to_num(
                            self.safe_lpips(fake_for_loss.float(), sharp_for_loss.float()),
                            nan=0.0
                        )
                        grad_loss = self.grad_loss(fake_for_loss, sharp_for_loss)
                        fft_sup_loss = fft_loss(fake_for_loss, sharp_for_loss)
                        edge_consistency = line_consistency_loss(fake_for_loss, sharp_for_loss)
                        centroid_loss = centroid_alignment_loss(fake_for_loss, sharp_for_loss)
                        
                        if warped is not None:
                            warp_loss = F.l1_loss(torch.clamp(warped, -1.0, 1.0), sharp_for_loss)
                            warp_psnr = self.psnr_metric(torch.clamp(warped, -1, 1), sharp_for_loss).item()
                            self.tb_logger.add_scalar("Debug/WarpPSNR", warp_psnr, global_step)

                        try:
                            psnr_val = self.psnr_metric(fake_for_loss, sharp_for_loss).item()
                            ssim_val = self.ssim_metric(fake_for_loss, sharp_for_loss).item()
                            lpips_val = float(torch.nan_to_num(
                                self.safe_lpips(fake_for_loss, sharp_for_loss), nan=0.0
                            ))
                            self.tb_logger.add_scalar("Train/PSNR", psnr_val, global_step)
                            self.tb_logger.add_scalar("Train/SSIM", ssim_val, global_step)
                            self.tb_logger.add_scalar("Train/LPIPS", lpips_val, global_step)
                            self.tb_logger.add_scalar("Loss/Warp", warp_loss.item(), global_step)
                        except Exception as e:
                            print(f"[WARN] Metric computation failed: {e}")

                    # Discriminator
                    featmatch_loss = torch.tensor(0.0, device=self.device)
                    adv_loss = torch.tensor(0.0, device=self.device)
                    loss_D = torch.tensor(0.0, device=self.device)
                    real_for_disc = None
                    
                    if epoch >= 150:
                        W_ADV_PATCH = 0.02   
                        W_ADV_FEATMATCH = 0.05                       
                        real_for_disc = torch.clamp(self._next_sharp(), -1.0, 1.0)
                        with torch.no_grad():
                            disc_out_real = self.discriminator(real_for_disc)
                            for key in ["score_full", "score_half", "score_quarter", "score_freq"]:
                                disc_out_real[key] = torch.nan_to_num(disc_out_real[key], nan=0.0)
                        
                        disc_out_fake = self.discriminator(fake_for_loss)
                        for key in ["score_full", "score_half", "score_quarter", "score_freq"]:
                            disc_out_fake[key] = torch.nan_to_num(disc_out_fake[key], nan=0.0)

                        featmatch_loss = self.discriminator.feature_matching_loss(
                            real_feats=disc_out_real["features"],
                            fake_feats=disc_out_fake["features"],
                        )
                        adv_loss_patch = -(
                            disc_out_fake["score_full"].mean() +
                            disc_out_fake["score_half"].mean() +
                            disc_out_fake["score_quarter"].mean()
                        )
                        adv_loss = W_ADV_PATCH * adv_loss_patch + W_ADV_FEATMATCH * featmatch_loss
                    
                    #losses                   
                    #phase 1: warmup, trajectory only
                    if epoch < 5:
                        if sharp_for_loss is not None and warped is not None:
                            total_loss = 2.0 * warp_loss + traj_smooth_loss + traj_align_loss
                        else:
                            total_loss = traj_smooth_loss + traj_align_loss
                    
                    elif epoch < 60:
                        if sharp_for_loss is not None and warped is not None:
                            total_loss = (2.0 * warp_loss +
                                          traj_smooth_loss +
                                          traj_align_loss +
                                          traj_magnitude_loss)
                        else:
                            total_loss = traj_smooth_loss + traj_align_loss + traj_magnitude_loss
                        total_loss = total_loss + W_FFT * freq_gain_loss
                    #phase 2+: decoder and full network
                    else:
                        total_loss = torch.tensor(0.0, device=self.device)
                        total_loss = total_loss + W_BLUR_PRIOR * blur_loss_val
                        total_loss = total_loss + W_PRIOR * prior_loss
                        total_loss = total_loss + W_BLUR_BAND * blur_band_val
                        total_loss = total_loss + W_DEPTH_LAYER * depth_layer_val
                        total_loss = total_loss + W_GHOST * ghost_edge_val
                        total_loss = total_loss + circ_weight * circ_loss
                        total_loss = total_loss + adv_loss
                        
                        if edge_map is not None and epoch >= 130:
                            total_loss = total_loss + W_EDGE_CONS * edge_loss
                        
                        if sharp_for_loss is not None:
                            total_loss = total_loss + 0.3 * W_L1 * l1_loss
                            total_loss = total_loss + 0.5 * lpips_loss
                            total_loss = total_loss + W_GRAD * grad_loss
                            total_loss = total_loss + W_FFT * fft_sup_loss
                            total_loss = total_loss + W_EDGE_CONS * edge_consistency
                            total_loss = total_loss + W_CENTROID * centroid_loss
                            total_loss = total_loss + 0.5 * warp_loss                  
                            #anti identity penalty
                            if epoch >= 60:
                                identity_sim = F.l1_loss(fake_for_loss, input_for_loss)
                                sharp_sim = F.l1_loss(fake_for_loss, sharp_for_loss)
                                total_loss = total_loss + 2.0 * F.relu(identity_sim - sharp_sim + 0.1)
                    
                    #backpropation
                    self.optimizer.G.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer.G)
                    #gradient boost for trajectory during phase 2 transition
                    if 60 <= epoch < 75:
                        for param in generator.trajectory_head.parameters():
                            if param.grad is not None:
                                param.grad *= 3.0

                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad),
                        max_norm=5.0
                    )
                    self.scaler.step(self.optimizer.G)
                    self.scaler.update()               
                    #discriminator update
                    if epoch >= 150 and idx % 2 == 0 and real_for_disc is not None:
                        self.optimizer_D.zero_grad()
                        with torch.no_grad():                   
                            disc_out_real_train = self.discriminator(real_for_disc)
                        disc_out_fake_train = self.discriminator(fake_for_loss.detach())
                        
                        loss_D_real = torch.tensor(0.0, device=self.device) 
                        for key in ["score_full", "score_half", "score_quarter"]:
                            loss_D_real += F.relu(1.0 - disc_out_real_train[key]).mean()
                        
                        loss_D_fake = torch.tensor(0.0, device=self.device)
                        for key in ["score_full", "score_half", "score_quarter"]:
                            loss_D_fake += F.relu(1.0 + disc_out_fake_train[key]).mean()
                            
                        loss_D = (loss_D_real + loss_D_fake) / 2.0
                            
                        self.scaler.scale(loss_D).backward()
                        self.scaler.step(self.optimizer_D)
                        self.scaler.update()

                    #logging
                    if idx % 50 == 0:
                        self.tb_logger.add_scalar("Loss/FFT_Gain", freq_gain_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Freq_Gain", freq_gain.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Blur_Prior", blur_loss_val.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Prior", prior_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/BlurBand", blur_band_val.item(), global_step)
                        self.tb_logger.add_scalar("Loss/DepthLayer", depth_layer_val.item(), global_step)
                        self.tb_logger.add_scalar("Loss/GhostEdge", ghost_edge_val.item(), global_step)
                        if depth_layer_info:
                            self.tb_logger.add_scalar("Loss/DL_FGSharp", depth_layer_info.get("fg_sharp", 0), global_step)
                            self.tb_logger.add_scalar("Loss/DL_BGSmooth", depth_layer_info.get("bg_smooth", 0), global_step)
                            self.tb_logger.add_scalar("Loss/DL_Boundary", depth_layer_info.get("boundary", 0), global_step)
                        self.tb_logger.add_scalar("Loss/Edge", edge_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/L1", l1_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/LPIPS", lpips_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Grad", grad_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/FFT_Sup", fft_sup_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/D", loss_D.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Total", total_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/CircularRefine", circ_loss.item(), global_step)                    
                        self.tb_logger.add_scalar("Loss/Adv", adv_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/FeatMatch", featmatch_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/TrajSmooth", traj_smooth_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/TrajAlign", traj_align_loss.item(), global_step)
                        
                        if trajectory is not None and dx_dy is not None:
                            traj_max = dx_dy.abs().max().item()
                            traj_mean = dx_dy.abs().mean().item()
                            self.tb_logger.add_scalar("Traj/Magnitude", traj_mean, global_step)
                            self.tb_logger.add_scalar("Traj/MaxDisp", traj_max, global_step)
                            if idx % 1052 == 0: 
                                resid_mag = (fake_for_loss - warped).abs().mean().item() if warped is not None else 0.0
                                print(f"[Traj] Epoch {epoch} Step {idx} | "
                                      f"MaxDisp={traj_max:.3f}  MeanDisp={traj_mean:.3f}  "
                                      f"WarpPSNR={warp_psnr:.2f}  TrainPSNR={psnr_val:.2f}  "
                                      f"ResidMag={resid_mag:.4f}")
                        self.tb_logger.flush()

                    log_line = (
                        f"Epoch {epoch}, Step {idx}, "
                        f"Total={total_loss.item():.4f}, D={loss_D.item():.4f}, "
                        f"L1={l1_loss.item():.4f}, FFT={fft_sup_loss.item():.4f}, "
                        f"BB={blur_band_val.item():.4f}, DL={depth_layer_val.item():.4f}, "
                        f"GE={ghost_edge_val.item():.4f}, CIRC={circ_loss.item():.4f}, "
                        f"CW={circ_weight:.4f}\n"
                    )
                    self.log_file.write(log_line)
                    self.log_file.flush()

                    tq.set_postfix(
                        loss=f"{total_loss.item():.3f}",
                        circ=f"{(circ_weight * circ_loss).item():.4f}",
                        lr=f"{self.optimizer.get_lr():.2e}",
                    )
                    tq.update(1)
                
                torch.cuda.empty_cache()
                self.optimizer.G.scheduler.step()

                #val.
                if (epoch + 1) % self.args.save_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for idx, batch in enumerate(self.loaders['val']):
                            blur = batch["blur"].to(self.device)
                            sharp = batch["sharp"].to(self.device) if batch["sharp"] is not None else None
                            relpath = batch["blur_path"]

                            output = self.model(blur)
                            output_fine = match_size(output["out"], sharp if sharp is not None else blur)
                            
                            output_vis = torch.clamp((output_fine + 1) / 2.0, 0, 1)
                            blur_vis = torch.clamp((blur + 1) / 2.0, 0, 1)

                            if sharp is not None:
                                sharp_vis = torch.clamp((sharp + 1) / 2.0, 0, 1)
                                triplet = torch.cat([blur_vis, output_vis, sharp_vis], dim=3)
                            else:
                                triplet = torch.cat([blur_vis, output_vis], dim=3)

                            base_name = os.path.basename(
                                relpath[0] if isinstance(relpath, (list, tuple)) else relpath
                            )
                            if idx < 3:
                                save_tensor_as_image(
                                    triplet[0],
                                    os.path.join(self.save_dir, f"val_epoch{epoch+1}_{base_name}.png")
                                )
                                if output.get("trajectory") is not None:
                                    save_trajectory_visualization(
                                        output["trajectory"][0],
                                        os.path.join(self.save_dir, f"traj_val_epoch{epoch+1}_{base_name}.png")
                                    )
                            else:
                                break
                    self.model.train()

                if (epoch + 1) % self.args.validate_every == 0:
                    print(f"[INFO] Running validation at epoch {epoch + 1}")
                    self.validate(epoch + 1)

                if self.args.rank == 0:
                    self.save(epoch)

    def evaluate(self, epoch, mode='val'):
        self.epoch = epoch
        self.model.eval()
        data_loader = self.loaders.get(mode, None)
        if data_loader is None:
            print(f"[WARNING] No data loader for mode '{mode}'")
            return

        paired_count = unpaired_count = 0
        psnr_vals, ssim_vals, lpips_vals = [], [], []
        niqe_vals, brisque_vals = [], []
        blur = output_fine = sharp = None
        base_name = "unknown"
        idx = -1

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                blur= batch["blur"].to(self.device)
                sharp = batch["sharp"].to(self.device) if batch["sharp"] is not None else None
                relpath = batch["blur_path"]

                output = self.model(blur)
                output_fine = match_size(output["out"], sharp if sharp is not None else blur)
                trajectory = output.get("trajectory", None)
                output_vis = torch.clamp((output_fine + 1) / 2.0, 0, 1)
                blur_vis = torch.clamp((blur+ 1) / 2.0, 0, 1)
                base_name = os.path.basename(relpath[0] if isinstance(relpath, (list, tuple)) else relpath)

                if sharp is not None:
                    sharp_vis = torch.clamp((sharp + 1) / 2.0, 0, 1)
                    triplet = torch.cat([blur_vis, output_vis, sharp_vis], dim=3)
                    paired_count += 1
                else:
                    triplet = torch.cat([blur_vis, output_vis], dim=3)
                    unpaired_count += 1

                if idx < 3:
                    save_tensor_as_image(
                        triplet[0],
                        os.path.join(self.save_dir, f"{mode}_epoch{epoch}_{base_name}.png")
                    )
                    if trajectory is not None:
                            save_trajectory_visualization(
                            trajectory[0],  # first image in batch
                            os.path.join(self.save_dir, f"traj_{mode}_epoch{epoch}_{base_name}.png")
                        )
                if sharp is not None:
                    try:
                        psnr_vals.append(self.psnr_metric(output_fine, sharp).item())
                        ssim_vals.append(self.ssim_metric(output_fine, sharp).item())
                        lpips_vals.append(float(
                            torch.nan_to_num(self.safe_lpips(output_fine, sharp), nan=0.0)
                        ))
                    except Exception as e:
                        print(f"[WARN] Metric failed for {base_name}: {e}")
                else:
                    try:
                        niqe_vals.append(self.niqe_metric(output_fine))
                        brisque_vals.append(self.brisque_metric(output_fine))
                    except Exception:
                        pass

        if psnr_vals:
            log_line = (
                f"[{mode.capitalize()} Epoch {epoch}] Paired N={len(psnr_vals)} "
                f"PSNR={sum(psnr_vals)/len(psnr_vals):.2f} "
                f"SSIM={sum(ssim_vals)/len(ssim_vals):.4f} "
                f"LPIPS={sum(lpips_vals)/len(lpips_vals):.4f}\n")
        else:
            log_line = f"[{mode.capitalize()} Epoch {epoch}] No paired reference metrics.\n"

        if niqe_vals:
            avg_brisque = sum(brisque_vals) / len(brisque_vals) if brisque_vals else float('nan')
            log_line += (
                f"[{mode.capitalize()} Epoch {epoch}] Unpaired N={len(niqe_vals)} "
                f"NIQE={sum(niqe_vals)/len(niqe_vals):.2f} BRISQUE={avg_brisque:.2f}\n")

        log_line += f"[INFO] Paired: {paired_count}, Unpaired: {unpaired_count}\n"
        self.log_file.write(log_line)
        print(log_line)
        self.log_file.flush()
 
        if self.args.rank == 0:
            self.save()

    def validate(self, epoch):
        self.evaluate(epoch, 'val')

    def test(self, epoch):
        self.evaluate(epoch, 'test')

    def fill_evaluation(self, epoch, mode=None, force=False):
        if epoch <= 0:
            return
        if mode is not None:
            self.mode = mode
        do_eval = force
        if not force:
            loss_missing = epoch not in self.criterion.loss_stat[self.mode]['Total']
            metric_missing = any(
                epoch not in self.criterion.metric_stat[mode][mt]
                for mt in self.criterion.metric
            )
            do_eval = loss_missing or metric_missing
        if do_eval:
            try:
                self.load(epoch)
                self.evaluate(epoch, self.mode)
            except Exception:
                pass

    def finish(self):
        if self.log_file:
            self.log_file.close()