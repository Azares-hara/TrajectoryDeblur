#Blur-to-Sharp Curriculum Learning for Motion Deblurring
#The model learns blur first, then gradually retracts to sharp
import os
import re
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
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
from lossfunc.metric import SSIMLoss
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
from torch.cuda.amp import GradScaler
from lossfunc.traildirection import BlurDirectionLoss


def line_consistency_loss(fake, ref):
    return F.l1_loss(sobel_edges(fake), sobel_edges(ref))


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff ** 2 + eps ** 2))


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


def phase_alignment_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    diff = torch.abs(pred_phase - target_phase)
    diff = torch.minimum(diff, 2 * torch.pi - diff)
    return diff.mean()


def dc_residual_loss(pred, target, blur_input, blur_prob=None, kernel_size=15):
    """
    Aura/ghosting suppression loss.
    Only penalizes residual in regions predicted as NON-BLUR (sharp content).
    """
    residual = blur_input - pred

    r_lp = F.avg_pool2d(residual.abs(), kernel_size, stride=1, padding=kernel_size//2)
    t_lp = F.avg_pool2d(target.abs(), kernel_size, stride=1, padding=kernel_size//2)

    if blur_prob is not None:
        sharp_mask = (1.0 - blur_prob).clamp(0.1, 1.0)
        rel_wide = (F.relu(r_lp - t_lp * 0.05) * sharp_mask).mean()
        abs_wide = (F.relu(r_lp - 0.01) * sharp_mask).mean()
    else:
        rel_wide = F.relu(r_lp - t_lp * 0.05).mean()
        abs_wide = F.relu(r_lp - 0.01).mean()

    r_fine = F.avg_pool2d(residual.abs(), 3, stride=1, padding=1)
    t_fine = F.avg_pool2d(target.abs(), 3, stride=1, padding=1)
    edge_mag = torch.abs(sobel_edges(target)).mean(dim=1, keepdim=True)
    edge_w = torch.clamp(edge_mag * 4.0, 0.5, 3.0)
    rel_fine = (F.relu(r_fine - t_fine * 0.02) * edge_w).mean()

    return rel_wide + abs_wide + 2.0 * rel_fine


def blur_schedule(epoch, total_epochs=400, warmup_epochs=50):
    """
    Curriculum schedule that reaches exactly 0.0 by epoch 365.
    Starts at 0.95 to prevent the 'do nothing' local minimum.
    """
    if epoch < warmup_epochs:
        # 0.95 -> 0.70 linearly (was 1.0 -> 0.7)
        return 0.95 - 0.25 * (epoch / warmup_epochs)
    elif epoch < 335:
        # 0.70 -> ~0.088 exponentially
        progress = (epoch - warmup_epochs) / (335 - warmup_epochs)
        return 0.70 * (0.5 ** (progress * 3))
    else:
        # Linear fade 0.088 -> 0.0 between epoch 335 and 365
        fade_progress = min(1.0, (epoch - 335) / 30.0)
        return 0.70 * (0.5 ** 3) * (1.0 - fade_progress)


def blur_consistency_loss(output, blur_input, trajectory, displacement_fn):
    """
    The model should be able to re-apply the predicted motion to its output,
    reconstructing the original blur. This proves it understands the blur process.
    """
    if trajectory is None or displacement_fn is None:
        return torch.tensor(0.0, device=output.device)
    reblurred = displacement_fn(output, trajectory)
    return F.l1_loss(reblurred, blur_input)


class TaskUncertainty(nn.Module):
    """
    Multi-task uncertainty weighting (Kendall & Gal).
    Automatically balances competing losses so gradients don't fight.
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """losses: list of scalar tensors."""
        total = 0.0
        for i, loss in enumerate(losses):
            inv_sigma = torch.exp(-self.log_sigma[i])
            total += inv_sigma * loss + self.log_sigma[i]
        return total


class EMA:
    def __init__(self, model, decay=0.999, shadow=None):
        self.decay = decay
        self.model = model
        if shadow is not None:
            self.shadow = shadow
        else:
            self.shadow = {}
            for k, v in model.named_parameters():
                if v.dtype.is_floating_point:
                    self.shadow[k] = v.clone().detach()
            for k, v in model.named_buffers():
                if v.dtype.is_floating_point:
                    self.shadow[k] = v.clone().detach()

    def update(self):
        for k, v in self.model.named_parameters():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()
        for k, v in self.model.named_buffers():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()

    def apply_shadow(self):
        self.backup = {}
        for k, v in self.model.named_parameters():
            if k in self.shadow:
                self.backup[k] = v.clone().detach()
                v.data.copy_(self.shadow[k])
        for k, v in self.model.named_buffers():
            if k in self.shadow:
                self.backup[k] = v.clone().detach()
                v.data.copy_(self.shadow[k])

    def restore(self):
        for k, v in self.model.named_parameters():
            if k in self.backup:
                v.data.copy_(self.backup[k])
        for k, v in self.model.named_buffers():
            if k in self.backup:
                v.data.copy_(self.backup[k])
        self.backup = {}


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

        num_workers = 4
        pin_memory = str(args.device).startswith('cuda')
        pf = 2 if num_workers > 0 else None

        self.loaders = {
            'train_blur':DataLoader(blur_dataset,  batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=True,
                              prefetch_factor=pf),
            'train_sharp': DataLoader(sharp_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=True,
                              prefetch_factor=pf),
            'val':         DataLoader(val_dataset,   batch_size=args.val_batch_size,
                              shuffle=False, num_workers=1,
                              pin_memory=pin_memory,
                              persistent_workers=False,
                              prefetch_factor=pf),
            'test':        DataLoader(test_dataset,  batch_size=args.val_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=False,
                              prefetch_factor=pf),
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
        self.epoch_psnr_sum = 0.0
        self.epoch_psnr_count = 0
        self.epoch_ssim_sum = 0.0
        self.epoch_ssim_count = 0
        self.epoch_lpips_sum = 0.0
        self.epoch_lpips_count = 0
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.epoch_l1_sum = 0.0
        self.epoch_l1_count = 0
        self.epoch_lpips_loss_sum = 0.0
        self.epoch_lpips_loss_count = 0
        self.epoch_ssim_loss_sum = 0.0
        self.epoch_ssim_loss_count = 0
        self.corner_mask = None

        # Discriminator setup (fixed: removed invalid n_scales argument)
        if hasattr(self.model, '_inner'):
            disc = self.model._inner('D')
            if disc is not None:
                self.discriminator = disc.to(self.device)
            else:
                self.discriminator = MultiScaleDiscriminator(n_feats=args.n_feats).to(self.device)
        else:
            if hasattr(self.model, 'model') and 'D' in self.model.model and self.model.model['D'] is not None:
                self.discriminator = self.model.model['D']
            else:
                self.discriminator = MultiScaleDiscriminator(n_feats=args.n_feats).to(self.device)

        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=args.lr_D, betas=(0.5, 0.999))

        # Loss modules
        self.freq_loss_module = FrequencyLoss(
            weight=args.lambda_freq,
            use_phase=True,
            multi_scale=True,
            scales=(2, 4),
            return_components=False,
        ).to(self.device)
        self.blur_band_loss   = BlurBandLoss(weight=0.05).to(self.device)
        self.depth_layer_loss = DepthLayerLoss(weight=0.05).to(self.device)
        self.ghost_edge_loss  = GhostEdgeLoss(weight=0.15, proximity_px=8).to(self.device)
        self.circular_loss = CircularBlurRefinementLoss(weight=1.0).to(self.device)
        self.blur_direction_loss = BlurDirectionLoss(weight=0.1, window=5).to(self.device)
        self.neighbor_loss = NeighborLoss()
        self.psnr_metric = PSNR(device=self.device)
        self.ssim_metric = SSIM(device_type=self.device.type)
        self.ssim_loss = SSIMLoss(channels=3).to(self.device)
        self.niqe_metric = NIQEMetric()
        self.brisque_metric = BRISQUEMetric()
        self.grad_loss = GradientLoss(device=self.device).to(torch.float32)
        self.lpips = get_lpips(net="squeeze", device=self.device, use_half=True)
        self.lpips_metric = self.safe_lpips

        # NEW: Task uncertainty balancer for 8 core losses
        # [L1, LPIPS, Grad, FFT_Sup, Phase, SSIM, EdgeCons, Aura]
        self.task_balancer = TaskUncertainty(num_tasks=8).to(self.device)
        self.optimizer_balancer = optim.Adam(self.task_balancer.parameters(), lr=1e-2)

        self.result_dir = (
            args.demo_output_dir if args.demo and args.demo_output_dir
            else os.path.join(args.save_dir, 'result')
        )
        os.makedirs(self.result_dir, exist_ok=True)
        print(f'Results are saved in {self.result_dir}')

        self.imsaver = MultiSaver(self.result_dir)
        self.scaler = GradScaler(init_scale=self.args.init_scale, enabled=self.args.amp)

        self.ema = None
        self.best_val_psnr = 0.0

        if self.args.resume:
            self._resume()
        else:
            self.ema = EMA(self.model, decay=0.999)

        log_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./"
        mode = "a" if self.args.resume else "w"
        self.log_file = open(os.path.join(log_dir, "train_log.txt"), mode)

    def validate_full(self, epoch):
        self.model.eval()
        psnr_vals, ssim_vals, lpips_vals = [], [], []
        with torch.no_grad():
            for batch in self.loaders['val']:
                blur = batch["blur"].to(self.device)
                sharp = batch["sharp"].to(self.device) if batch["sharp"] is not None else None
                if sharp is None:
                    continue
                output = self.model(blur)
                out = match_size(output["out"], sharp)
                psnr_vals.append(self.psnr_metric(out, sharp).item())
                ssim_vals.append(self.ssim_metric(out, sharp).item())
                lpips_vals.append(float(torch.nan_to_num(
                    self.safe_lpips(out, sharp), nan=0.0)))
        avg_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0.0
        avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0.0
        avg_lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0
        print(f"[VAL FULL Epoch {epoch}] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")
        self.tb_logger.add_scalar("ValFull/PSNR", avg_psnr, epoch)
        self.tb_logger.add_scalar("ValFull/SSIM", avg_ssim, epoch)
        self.tb_logger.add_scalar("ValFull/LPIPS", avg_lpips, epoch)
        self.model.train()
        return avg_psnr

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
                return
            print(f"[WARNING] Manual epoch {epoch} not found. Falling back to auto-resume.")
            self.epoch = 1
            return
        if not os.path.exists(model_dir):
            print("[WARNING] No models directory found. Starting from scratch.")
            self.epoch = 1
            self.ema = EMA(self.model, decay=0.999)
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
                    opt_obj.scheduler.last_epoch = -1
                    opt_obj.scheduler.step(0)
                    print(f"[INFO] Reset {opt_name} scheduler to base LR: "
                      f"{opt_obj.param_groups[0]['lr']:.2e}")
        else:
            print("[WARNING] No valid checkpoint found. Starting from scratch.")
            self.epoch = 1
            self.ema = EMA(self.model, decay=0.999)

    def save(self, epoch=None, is_best=False):
        epoch = self.epoch if epoch is None else epoch
        should_save_regular = (epoch % self.args.save_every == 0)
        if should_save_regular:
            self.model.save(epoch)
            self.optimizer.save(epoch)
            if self.ema is not None:
                ema_path = os.path.join(self.args.save_dir, 'models', f"ema-{epoch}.pt")
                torch.save(self.ema.shadow, ema_path)
            torch.save(self.optimizer_D.state_dict(),
                       os.path.join(self.args.save_dir, 'optim', f"optim_D-{epoch}.pt"))
            # Save task balancer state
            bal_path = os.path.join(self.args.save_dir, 'optim', f"balancer-{epoch}.pt")
            torch.save(self.task_balancer.state_dict(), bal_path)
        if is_best and not should_save_regular:
            self.model.save(epoch)
            self.optimizer.save(epoch)
            if self.ema is not None:
                ema_path = os.path.join(self.args.save_dir, 'models', f"ema-{epoch}.pt")
                torch.save(self.ema.shadow, ema_path)
            torch.save(self.optimizer_D.state_dict(),os.path.join(self.args.save_dir, 'optim', f"optim_D-{epoch}.pt"))
            bal_path = os.path.join(self.args.save_dir, 'optim', f"balancer-{epoch}.pt")
            torch.save(self.task_balancer.state_dict(), bal_path)
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'models', "model-best.pt")
            torch.save(self.model.state_dict(), best_path)
            print(f"[INFO] Saved best model with PSNR {self.best_val_psnr:.2f}")

    def load(self, epoch=None, pretrained=None):
        if epoch is None:
            epoch = self.args.load_epoch
        self.model.load(epoch, pretrained)
        self.optimizer.load(epoch)

        ema_path = os.path.join(self.args.save_dir, 'models', f"ema-{epoch}.pt")
        loaded_shadow = None

        if os.path.exists(ema_path):
            print(f"[INFO] Loading EMA shadow from {ema_path}")
            loaded_shadow = torch.load(ema_path, map_location=self.device, weights_only=False)
        else:
            old_ema_path = os.path.join(self.args.resume_dir, 'models', f"ema-{epoch}.pt") if hasattr(self.args, 'resume_dir') else ""
            if old_ema_path and os.path.exists(old_ema_path):
                print(f"[INFO] Loading EMA shadow from {old_ema_path}")
                loaded_shadow = torch.load(old_ema_path, map_location=self.device, weights_only=False)
            else:
                print(f"[WARNING] No EMA checkpoint found; re-initializing EMA from current weights.")

        if loaded_shadow is not None:
            current_state = {}
            for module_name, module_sd in self.model.state_dict().items():
                if isinstance(module_sd, dict):
                    for k, v in module_sd.items():
                        if hasattr(v, 'dtype') and v.dtype.is_floating_point:
                            current_state[f"{module_name}.{k}"] = v
                elif hasattr(module_sd, 'dtype') and module_sd.dtype.is_floating_point:
                    current_state[module_name] = module_sd

            valid_shadow = {}
            missing_in_shadow = []
            for k, v in loaded_shadow.items():
                if k in current_state:
                    valid_shadow[k] = v.to(device=self.device, dtype=current_state[k].dtype)
                else:
                    missing_in_shadow.append(k)

            if missing_in_shadow:
                print(f"[INFO] {len(missing_in_shadow)} EMA keys from checkpoint not in current model "
                      f"(new layers expected): {missing_in_shadow[:5]}{'...' if len(missing_in_shadow) > 5 else ''}")

            for k, v in current_state.items():
                if k not in valid_shadow:
                    valid_shadow[k] = v.clone().detach()
                    print(f"[INFO] EMA key {k} initialized from current weights (new layer)")

            bad_ema_keys = [k for k, v in valid_shadow.items() if not torch.isfinite(v).all()]
            if bad_ema_keys:
                print(f"[WARN] {len(bad_ema_keys)} EMA keys have NaN/Inf, re-initializing from current weights")
                for k in bad_ema_keys:
                    valid_shadow[k] = current_state[k].clone().detach()

            self.ema = EMA(self.model, decay=0.999, shadow=valid_shadow)
            print(f"[INFO] EMA loaded with {len(valid_shadow)} keys")
        else:
            self.ema = EMA(self.model, decay=0.999)

        ckpt_D = os.path.join(self.args.save_dir, 'optim', f"optim_D-{epoch}.pt")
        if os.path.exists(ckpt_D):
            print(f"[INFO] Loading discriminator optimizer from {ckpt_D}")
            self.optimizer_D.load_state_dict(torch.load(ckpt_D, map_location=self.device, weights_only=False))
        else:
            print(f"[WARNING] No discriminator optimizer checkpoint at {ckpt_D}")

        # Load task balancer
        bal_path = os.path.join(self.args.save_dir, 'optim', f"balancer-{epoch}.pt")
        if os.path.exists(bal_path):
            print(f"[INFO] Loading task balancer from {bal_path}")
            self.task_balancer.load_state_dict(torch.load(bal_path, map_location=self.device, weights_only=False))
        else:
            print(f"[WARNING] No task balancer checkpoint at {bal_path}")

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
        ACCUM_STEPS = 2

        # Base weights (will be overridden by task balancer for core losses)
        BASE_W_BLUR_PRIOR = 0.01
        BASE_W_PRIOR = 0.005
        BASE_W_BLUR_BAND = 0.01
        BASE_W_DEPTH_LAYER = 0.01
        BASE_W_GHOST = 0.3
        BASE_W_GRAD = 0.2
        BASE_W_FFT = 1.0
        BASE_W_LPIPS = 1.0
        BASE_W_EDGE_CONS = 0.8
        BASE_W_CENTROID = 0.0
        BASE_W_L1 = 0.6
        BASE_W_SSIM = 0.8
        BASE_W_PHASE = 0.4
        BASE_W_AURA = 0.8

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            # BLUR CURRICULUM
            blur_alpha = blur_schedule(epoch, total_epochs=num_epochs, warmup_epochs=50)

            # Ramping schedule for late-stage losses
            if epoch <= 335:
                ramp = 0.0
            elif epoch >= 365:
                ramp = 1.0
            else:
                ramp = (epoch - 335) / 30.0

            W_GHOST = 0.3 + 0.4 * ramp
            W_AURA = 0.8 + 0.4 * ramp
            W_EDGE_CONS = 0.8 + 0.2 * ramp
            W_PHASE = 0.4 + 0.1 * ramp

            W_BLUR_PRIOR = BASE_W_BLUR_PRIOR
            W_PRIOR = BASE_W_PRIOR
            W_BLUR_BAND = BASE_W_BLUR_BAND
            W_DEPTH_LAYER = BASE_W_DEPTH_LAYER
            W_GRAD = BASE_W_GRAD
            W_FFT = BASE_W_FFT
            W_LPIPS = BASE_W_LPIPS
            W_CENTROID = BASE_W_CENTROID
            W_L1 = BASE_W_L1
            W_SSIM = BASE_W_SSIM

            # NEW: Late-stage PSNR boost — reduce perceptual, increase pixel/structural
            if epoch >= 350:
                W_LPIPS *= 0.5
                W_L1 *= 1.2
                W_SSIM *= 1.2
                W_GRAD *= 0.8

            if 366 <= epoch <= 370:
                for param_group in self.optimizer.G.param_groups:
                    param_group['lr'] = 5e-5

            generator = self.model._inner('G') if hasattr(self.model, '_inner') else self.model
            for param in generator.parameters():
                param.requires_grad = True

            with tqdm(total=len(self.loaders['train_blur']), ncols=100,
                      desc=f"Epoch {self.epoch} blurα={blur_alpha:.2f}") as tq:
                self.optimizer.G.zero_grad()
                self.optimizer_balancer.zero_grad()

                for idx, batch in enumerate(self.real_loader):
                    global_step = epoch * len(self.loaders['train_blur']) + idx

                    real_blur = batch["blur"]
                    real_sharp = batch["sharp"]
                    input_real, _ = dataset.common.to(
                        real_blur, None, device=self.device, dtype=self.dtype_eval
                    )
                    input_real = input_real.to(self.device, non_blocking=True)
                    outputs = self.model(input_real, stage)

                    fake_real = outputs["out"]
                    warped = outputs.get("warped")
                    trajectory = outputs.get("trajectory", None)
                    fake_real = match_size(fake_real, input_real)
                    blur_prob = outputs.get("blur_prob", None)
                    edge_map = outputs.get("edge_map", None)

                    # Aggressive NaN/Inf sanitization (defensive, not intrusive)
                    fake_real = torch.nan_to_num(fake_real, nan=0.0, posinf=1.0, neginf=-1.0)
                    if blur_prob is not None:
                        blur_prob = torch.nan_to_num(blur_prob, nan=0.0, posinf=1.0, neginf=0.0)
                    if edge_map is not None:
                        edge_map = torch.nan_to_num(edge_map, nan=0.0, posinf=1.0, neginf=0.0)
                    if trajectory is not None:
                        trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)

                    sharp_real = (
                        real_sharp.to(self.device, dtype=self.dtype_eval, non_blocking=True)
                        if real_sharp is not None else None
                    )

                    fake_for_loss  = torch.clamp(fake_real,  -1.0, 1.0)
                    input_for_loss = torch.clamp(input_real, -1.0, 1.0)
                    sharp_for_loss = (
                        torch.clamp(sharp_real, -1.0, 1.0) if sharp_real is not None else None
                    )

                    B, C, H, W = fake_for_loss.shape
                    if self.corner_mask is None or self.corner_mask.shape[-2:] != (H, W):
                        y = torch.arange(H, device=fake_for_loss.device).float().view(1, 1, H, 1)
                        x = torch.arange(W, device=fake_for_loss.device).float().view(1, 1, 1, W)
                        cy, cx = H / 2.0, W / 2.0
                        dist = torch.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
                        self.corner_mask = (dist > 0.5).float() * 2.0 + 1.0

                    curriculum_target = None
                    if sharp_for_loss is not None:
                        curriculum_target = blur_alpha * input_for_loss + (1 - blur_alpha) * sharp_for_loss

                    curriculum_output = blur_alpha * input_for_loss + (1 - blur_alpha) * fake_for_loss

                    # Frequency gain (sharpening incentive for blurry regions)
                    freq_gain_loss = torch.tensor(0.0, device=self.device)
                    freq_gain_val = 0.0
                    if blur_prob is not None:
                        with torch.no_grad():
                            fft_fake = torch.fft.rfft2(fake_for_loss.float(), norm="ortho")
                            fft_input = torch.fft.rfft2(input_for_loss.float(), norm="ortho")
                            freq_gain = (torch.mean(torch.abs(fft_fake)) - torch.mean(torch.abs(fft_input)))
                        freq_gain_val = freq_gain.item() if torch.isfinite(freq_gain).all() else float('nan')
                        if torch.isfinite(freq_gain) and torch.abs(freq_gain) < 5.0:
                            freq_gain_clamped = torch.clamp(freq_gain, min=-3.0, max=3.0)
                            # Positive incentive: boost frequencies in blurry regions
                            freq_gain_loss = -0.005 * blur_prob.mean() * freq_gain_clamped

                    # Unsupervised losses
                    blur_loss_val = torch.tensor(0.0, device=self.device)
                    prior_loss = torch.tensor(0.0, device=self.device)
                    if blur_prob is not None and epoch >= 50:
                        blur_loss_val = compute_blur_loss(blur_prob, fake_for_loss)
                        prior_loss = sharpness_prior_loss(blur_prob, fake_real, input_real)

                    blur_band_val = torch.tensor(0.0, device=self.device)
                    if epoch >= 10:
                        blur_band_val = self.blur_band_loss(fake_for_loss, input_for_loss)

                    depth_layer_val = torch.tensor(0.0, device=self.device)
                    depth_layer_info = {}
                    if epoch >= 50:
                        depth_layer_val, depth_layer_info = self.depth_layer_loss(
                            fake_for_loss, input_for_loss, blur_prob=blur_prob)

                    ghost_edge_val = torch.tensor(0.0, device=self.device)
                    if epoch >= 150:
                        ghost_edge_val = self.ghost_edge_loss(fake_for_loss, input_for_loss)

                    edge_loss = torch.tensor(0.0, device=self.device)
                    if edge_map is not None and epoch >= 130:
                        target_edges = sobel_edges(fake_for_loss)
                        target_edges_min = target_edges.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
                        target_edges_max = target_edges.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
                        target_edges_norm = (target_edges - target_edges_min) / (target_edges_max - target_edges_min + 1e-6)
                        edge_loss = torch.nan_to_num(
                            F.l1_loss(edge_map, target_edges_norm),
                            nan=0.0, posinf=1.0, neginf=-1.0
                        )

                    circ_loss = torch.tensor(0.0, device=self.device)
                    circ_weight = 0.0
                    if epoch >= 140:
                        circ_weight = max(0.0, min(1.0, 0.1 + (epoch - 150) * 0.02))
                        circ_loss = self.circular_loss(fake_for_loss, input_for_loss, blur_prob)

                    # Supervised losses (will be fed into task balancer)
                    l1_loss = torch.tensor(0.0, device=self.device)
                    lpips_loss = torch.tensor(0.0, device=self.device)
                    grad_loss = torch.tensor(0.0, device=self.device)
                    phase_loss_val = torch.tensor(0.0, device=self.device)
                    fft_sup_loss = torch.tensor(0.0, device=self.device)
                    edge_consistency = torch.tensor(0.0, device=self.device)
                    centroid_loss = torch.tensor(0.0, device=self.device)
                    ssim_loss_val = torch.tensor(0.0, device=self.device)
                    aura_loss = torch.tensor(0.0, device=self.device)
                    blur_consistency = torch.tensor(0.0, device=self.device)

                    if curriculum_target is not None:
                        base_l1 = F.l1_loss(curriculum_output, curriculum_target)
                        weighted_l1 = (F.l1_loss(curriculum_output, curriculum_target, reduction='none') * self.corner_mask).mean()
                        charb = charbonnier_loss(curriculum_output, curriculum_target)

                        l1_loss = 0.33 * base_l1 + 0.33 * weighted_l1 + 0.34 * charb
                        if blur_prob is not None:
                            uncertainty = blur_prob.detach()
                            l1_err = F.l1_loss(curriculum_output, curriculum_target, reduction='none')
                            l1_loss = l1_loss + (l1_err * (2.0 * uncertainty)).mean()

                        fft_sup_loss = self.freq_loss_module(curriculum_output, curriculum_target)
                        phase_loss_val = phase_alignment_loss(curriculum_output, curriculum_target)

                        fake_edges = sobel_edges(curriculum_output)
                        sharp_edges = sobel_edges(curriculum_target)
                        edge_err = torch.abs(fake_edges - sharp_edges)
                        edge_consistency = (edge_err * self.corner_mask).mean()

                        aura_loss = dc_residual_loss(curriculum_output, curriculum_target, input_for_loss, blur_prob=blur_prob)
                        centroid_loss = centroid_alignment_loss(curriculum_output, curriculum_target)

                        ssim_loss_val = self.ssim_loss((curriculum_output + 1) / 2.0, (curriculum_target + 1) / 2.0)
                        lpips_loss = torch.nan_to_num(self.safe_lpips(curriculum_output.float(), curriculum_target.float()), nan=0.0)
                        grad_loss = self.grad_loss(curriculum_output, curriculum_target)

                        # Blur consistency (understanding of blur process)
                        if trajectory is not None and hasattr(generator, 'displacement'):
                            blur_consistency = blur_consistency_loss(
                                fake_for_loss, input_for_loss, trajectory, generator.displacement
                            )

                        if idx % 50 == 0:
                            try:
                                psnr_val = self.psnr_metric(curriculum_output, curriculum_target).item()
                                ssim_val = self.ssim_metric(curriculum_output, curriculum_target).item()
                                lpips_val = float(torch.nan_to_num(self.safe_lpips(curriculum_output, curriculum_target), nan=0.0))
                                self.tb_logger.add_scalar("Train/PSNR", psnr_val, global_step)
                                self.tb_logger.add_scalar("Train/SSIM", ssim_val, global_step)
                                self.tb_logger.add_scalar("Train/LPIPS", lpips_val, global_step)
                                self.epoch_psnr_sum += psnr_val
                                self.epoch_psnr_count += 1
                                self.epoch_ssim_sum += ssim_val
                                self.epoch_ssim_count += 1
                                self.epoch_lpips_sum += lpips_val
                                self.epoch_lpips_count += 1
                            except Exception as e:
                                print(f"[WARN] Metric computation failed: {e}")

                    # Discriminator
                    featmatch_loss = torch.tensor(0.0, device=self.device)
                    adv_loss = torch.tensor(0.0, device=self.device)
                    loss_D = torch.tensor(0.0, device=self.device)
                    real_for_disc = None
                    adv_start, adv_full = 30, 100
                    w_adv = 0.0
                    if epoch >= adv_start:
                        w_adv = min(1.0, (epoch - adv_start) / (adv_full - adv_start)) * 0.02
                    if w_adv > 0:
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
                        adv_loss_patch = -(disc_out_fake["score_full"].mean() +
                                          disc_out_fake["score_half"].mean() +
                                          disc_out_fake["score_quarter"].mean() +
                                          disc_out_fake["score_freq"].mean())
                        adv_loss = W_ADV_PATCH * adv_loss_patch + W_ADV_FEATMATCH * featmatch_loss

                    # === NEW: Task uncertainty balancing for core losses ===
                    # Order: [L1, LPIPS, Grad, FFT_Sup, Phase, SSIM, EdgeCons, Aura]
                    core_losses = [
                        l1_loss,
                        lpips_loss,
                        grad_loss,
                        fft_sup_loss,
                        phase_loss_val,
                        ssim_loss_val,
                        edge_consistency,
                        aura_loss,
                    ]
                    core_total = self.task_balancer(core_losses)

                    # Auxiliary losses (fixed manual weights)
                    total_loss = core_total
                    total_loss = total_loss + freq_gain_loss
                    total_loss = total_loss + W_BLUR_PRIOR * blur_loss_val
                    total_loss = total_loss + W_PRIOR * prior_loss
                    total_loss = total_loss + W_AURA * aura_loss  # note: aura is in core too, but this adds extra emphasis
                    total_loss = total_loss + blur_alpha * blur_consistency

                    if epoch >= 10:
                        total_loss = total_loss + W_BLUR_BAND * blur_band_val
                    if epoch >= 50:
                        total_loss = total_loss + W_DEPTH_LAYER * depth_layer_val

                    # Trajectory regularization
                    if trajectory is not None and epoch >= 50:
                        dx_dy = trajectory[:, :2]
                        conf = trajectory[:, 2:3]
                        conf_target = blur_prob.detach() if blur_prob is not None else torch.zeros_like(conf)
                        conf_loss = F.l1_loss(conf, conf_target)
                        total_loss = total_loss + 0.1 * conf_loss
                        motion_norm = torch.sqrt(dx_dy[:, 0:1]**2 + dx_dy[:, 1:2]**2 + 1e-6)
                        pred_motion = dx_dy / (motion_norm + 1e-6)
                        if blur_prob is not None:
                            blur_dir_loss = self.blur_direction_loss(input_for_loss, pred_motion, blur_prob)
                            total_loss = total_loss + 0.1 * blur_dir_loss
                        dx_h = dx_dy[:, :, :, 1:] - dx_dy[:, :, :, :-1]
                        dx_v = dx_dy[:, :, 1:, :] - dx_dy[:, :, :-1, :]
                        traj_smooth = (dx_h.abs().mean() + dx_v.abs().mean()) * 0.05
                        traj_conf = conf.mean() * 0.005
                        total_loss = total_loss + traj_smooth + traj_conf

                    if sharp_for_loss is not None:
                        identity_sim = F.l1_loss(fake_for_loss, input_for_loss)
                        sharp_sim = F.l1_loss(fake_for_loss, sharp_for_loss)
                        total_loss = total_loss + 1.0 * F.relu(sharp_sim - identity_sim + 0.05)

                    if w_adv > 0:
                        total_loss = total_loss + w_adv * adv_loss

                    if epoch >= 150:
                        total_loss = total_loss + circ_weight * circ_loss
                        total_loss = total_loss + W_GHOST * ghost_edge_val

                    if edge_map is not None and epoch >= 130:
                        edge_loss_weight = 0.1 + 0.4 * ramp
                        total_loss = total_loss + edge_loss_weight * edge_loss

                    if warped is not None and sharp_for_loss is not None:
                        aux_warp_weight = 0.1 + 0.3 * ramp
                        aux_warp = F.l1_loss(warped, sharp_for_loss)
                        total_loss = total_loss + aux_warp_weight * aux_warp

                    total_loss = total_loss / ACCUM_STEPS

                    # Generator update
                    self.scaler.scale(total_loss).backward()

                    if (idx + 1) % ACCUM_STEPS == 0:
                        self.scaler.unscale_(self.optimizer.G)
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            max_norm=5.0
                        )
                        self.scaler.step(self.optimizer.G)
                        # EMA updates ONLY when weights actually change (correct placement)
                        if self.ema is not None:
                            self.ema.update()
                        # Step task balancer (tiny LR, stable)
                        self.scaler.step(self.optimizer_balancer)
                        self.scaler.update()
                        self.optimizer.G.zero_grad()
                        self.optimizer_balancer.zero_grad()

                    # Discriminator in pure FP32
                    if w_adv > 0 and idx % 2 == 0 and real_for_disc is not None:
                        self.optimizer_D.zero_grad()
                        disc_out_real_train = self.discriminator(real_for_disc.float())
                        disc_out_fake_train = self.discriminator(fake_for_loss.detach().float())

                        loss_D_real = torch.tensor(0.0, device=self.device)
                        skip_D = False
                        for key in ["score_full", "score_half", "score_quarter", "score_freq"]:
                            val = disc_out_real_train[key]
                            if torch.isnan(val).any() or torch.isinf(val).any():
                                print(f"[WARN] D real {key} has NaN/Inf, skipping D update")
                                skip_D = True
                                break
                            loss_D_real += F.relu(1.0 - val).mean()

                        if not skip_D:
                            loss_D_fake = torch.tensor(0.0, device=self.device)
                            for key in ["score_full", "score_half", "score_quarter", "score_freq"]:
                                val = disc_out_fake_train[key]
                                if torch.isnan(val).any() or torch.isinf(val).any():
                                    print(f"[WARN] D fake {key} has NaN/Inf, skipping D update")
                                    skip_D = True
                                    break
                                loss_D_fake += F.relu(1.0 + val).mean()

                        if not skip_D:
                            loss_D = (loss_D_real + loss_D_fake) / 2.0
                            if torch.isfinite(loss_D):
                                loss_D.backward()
                                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
                                self.optimizer_D.step()
                            else:
                                print(f"[WARN] D loss is NaN/Inf, skipping D update")

                    # Logging
                    if idx % 50 == 0:
                        self.tb_logger.add_scalar("Curriculum/BlurAlpha", blur_alpha, global_step)
                        self.tb_logger.add_scalar("Curriculum/BlurConsistency", blur_consistency.item(), global_step)
                        self.tb_logger.add_scalar("Loss/FFT_Gain", freq_gain_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Freq_Gain", freq_gain_val, global_step)
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
                        self.tb_logger.add_scalar("Loss/Total", total_loss.item() * ACCUM_STEPS, global_step)
                        self.tb_logger.add_scalar("Loss/CircularRefine", circ_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Adv", adv_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/FeatMatch", featmatch_loss.item(), global_step)
                        self.tb_logger.add_scalar("Loss/SSIM", ssim_loss_val.item(), global_step)
                        self.tb_logger.add_scalar("Loss/Aura", aura_loss.item(), global_step)
                        self.tb_logger.add_scalar("Weights/W_GHOST", W_GHOST, global_step)
                        self.tb_logger.add_scalar("Weights/W_AURA", W_AURA, global_step)
                        self.tb_logger.add_scalar("Weights/W_EDGE_CONS", W_EDGE_CONS, global_step)
                        self.tb_logger.add_scalar("Weights/W_PHASE", W_PHASE, global_step)
                        self.tb_logger.add_scalar("Weights/Ramp", ramp, global_step)
                        # Log task balancer sigmas
                        for i, name in enumerate(["L1","LPIPS","Grad","FFT","Phase","SSIM","Edge","Aura"]:
                            self.tb_logger.add_scalar(f"Balancer/sigma_{name}", torch.exp(self.task_balancer.log_sigma[i]).item(), global_step)

                        self.epoch_loss_sum += total_loss.item() * ACCUM_STEPS
                        self.epoch_loss_count += 1
                        self.epoch_l1_sum += l1_loss.item()
                        self.epoch_l1_count += 1
                        self.epoch_lpips_loss_sum += lpips_loss.item()
                        self.epoch_lpips_loss_count += 1
                        self.epoch_ssim_loss_sum += ssim_loss_val.item()
                        self.epoch_ssim_loss_count += 1

                        if blur_prob is not None:
                            self.tb_logger.add_histogram("BlurHead/Prob", blur_prob, global_step)
                        if trajectory is not None:
                            traj_mag = trajectory[:, :2].abs().mean()
                            self.tb_logger.add_scalar("Traj/MeanMag", traj_mag.item(), global_step)
                            self.tb_logger.add_scalar("Traj/MeanConf", trajectory[:, 2:3].mean().item(), global_step)
                        self.tb_logger.flush()

                        if idx % 500 == 0 and sharp_for_loss is not None:
                            vis_blur  = (input_for_loss[0] + 1) / 2.0
                            vis_out   = (fake_for_loss[0] + 1) / 2.0
                            vis_sharp = (sharp_for_loss[0] + 1) / 2.0
                            triplet = torch.cat([vis_blur, vis_out, vis_sharp], dim=1).detach().cpu()
                            self.tb_logger.add_image("Train/Triplet", triplet.clamp(0, 1), global_step)

                    log_line = (
                        f"Epoch {epoch}, Step {idx}, "
                        f"blurα={blur_alpha:.3f}, Ramp={ramp:.2f}, "
                        f"Total={total_loss.item() * ACCUM_STEPS:.4f}, D={loss_D.item():.4f}, "
                        f"L1={l1_loss.item():.4f}, FFT={fft_sup_loss.item():.4f}, "
                        f"BB={blur_band_val.item():.4f}, DL={depth_layer_val.item():.4f}, "
                        f"GE={ghost_edge_val.item():.4f}, CIRC={circ_loss.item():.4f}, "
                        f"Aura={aura_loss.item():.4f}, BlurCons={blur_consistency.item():.4f}, "
                        f"CW={circ_weight:.4f}
"
                    )
                    self.log_file.write(log_line)
                    self.log_file.flush()

                    tq.set_postfix(
                        loss=f"{total_loss.item() * ACCUM_STEPS:.3f}",
                        blur=f"{blur_alpha:.2f}",
                        circ=f"{(circ_weight * circ_loss).item():.4f}",
                        lr=f"{self.optimizer.get_lr():.2e}",
                    )
                    tq.update(1)

                # Epoch-end logging
                if self.epoch_psnr_count > 0:
                    self.tb_logger.add_scalar("Epoch/PSNR", self.epoch_psnr_sum / self.epoch_psnr_count, self.epoch)
                    self.tb_logger.add_scalar("Epoch/SSIM", self.epoch_ssim_sum / self.epoch_ssim_count, self.epoch)
                    self.tb_logger.add_scalar("Epoch/LPIPS", self.epoch_lpips_sum / self.epoch_lpips_count, self.epoch)

                if self.epoch_loss_count > 0:
                    self.tb_logger.add_scalar("Epoch/Loss_Total", self.epoch_loss_sum / self.epoch_loss_count, self.epoch)
                    self.tb_logger.add_scalar("Epoch/Loss_L1", self.epoch_l1_sum / self.epoch_l1_count, self.epoch)
                    self.tb_logger.add_scalar("Epoch/Loss_LPIPS", self.epoch_lpips_loss_sum / self.epoch_lpips_loss_count, self.epoch)
                    self.tb_logger.add_scalar("Epoch/Loss_SSIM", self.epoch_ssim_loss_sum / self.epoch_ssim_loss_count, self.epoch)
                    self.epoch_loss_sum = 0.0
                    self.epoch_loss_count = 0
                    self.epoch_l1_sum = 0.0
                    self.epoch_l1_count = 0
                    self.epoch_lpips_loss_sum = 0.0
                    self.epoch_lpips_loss_count = 0
                    self.epoch_ssim_loss_sum = 0.0
                    self.epoch_ssim_loss_count = 0

                self.epoch_psnr_sum = 0.0
                self.epoch_psnr_count = 0
                self.epoch_ssim_sum = 0.0
                self.epoch_ssim_count = 0
                self.epoch_lpips_sum = 0.0
                self.epoch_lpips_count = 0
                torch.cuda.empty_cache()
                self.optimizer.G.scheduler.step()

                # === Visual sample saving (separate from quantitative validation) ===
                if (epoch + 1) % self.args.save_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for idx, batch in enumerate(self.loaders['val']):
                            if idx >= 3:
                                break  # Only save first 3 images
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

                            base_name = os.path.basename(relpath[0] if isinstance(relpath, (list, tuple)) else relpath)
                            save_tensor_as_image(triplet[0], os.path.join(self.result_dir, f"val_epoch{epoch+1}_{base_name}.png"))
                    self.model.train()

                # === Full quantitative validation ===
                if (epoch + 1) % self.args.validate_every == 0:
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    current_psnr = self.validate_full(epoch + 1)
                    if self.ema is not None:
                        self.ema.restore()
                    self.model.train()
                    if current_psnr > self.best_val_psnr:
                        self.best_val_psnr = current_psnr
                        self.save(epoch=epoch+1, is_best=True)

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
                    save_tensor_as_image(triplet[0],os.path.join(self.result_dir, f"{mode}_epoch{epoch}_{base_name}.png"))

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
                f"LPIPS={sum(lpips_vals)/len(lpips_vals):.4f}
")
        else:
            log_line = f"[{mode.capitalize()} Epoch {epoch}] No paired reference metrics.
"

        if niqe_vals:
            avg_brisque = sum(brisque_vals) / len(brisque_vals) if brisque_vals else float('nan')
            log_line += (
                f"[{mode.capitalize()} Epoch {epoch}] Unpaired N={len(niqe_vals)} "
                f"NIQE={sum(niqe_vals)/len(niqe_vals):.2f} BRISQUE={avg_brisque:.2f}
")

        log_line += f"[INFO] Paired: {paired_count}, Unpaired: {unpaired_count}
"
        self.log_file.write(log_line)
        print(log_line)
        self.log_file.flush()

        if self.args.rank == 0:
            self.save()

        self.model.train()

    def validate(self, epoch):
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        self.evaluate(epoch, 'val')
        if self.ema is not None:
            self.ema.restore()

    def test(self, epoch):
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        self.evaluate(epoch, 'test')
        if self.ema is not None:
            self.ema.restore()

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
