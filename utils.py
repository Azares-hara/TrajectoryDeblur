import code
import pdb
import time
import argparse
import os
import imageio
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
try:
    import readline
    import rlcompleter
    readline.parse_and_bind("tab: complete")
except ImportError:
    print("[WARNING] readline not available - autocomplete disabled.")

if __name__ != "__main__" and 'ipykernel' in sys.modules:
    print("[WARNING] Multiprocessing may not behave as expected in Jupyter.")

def save_trajectory_visualization(trajectory, save_path):
    """
    trajectory: Tensor of shape (3, H, W) or (B, 3, H, W) from your model.
                Expects [dx, dy, conf] in channels 0,1,2.
    """
    if trajectory.ndim == 4:
        trajectory = trajectory[0]
    
    traj_np = trajectory.detach().cpu().numpy()  # (3, H, W)
    dx, dy = traj_np[0], traj_np[1]
    
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    #HSV: Hue = angle, Saturation = 1, Value = magnitude
    hsv = np.zeros((*dx.shape, 3))
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # normalize to [0,1]
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(magnitude / (magnitude.mean() + 1e-6), 0, 1)
    
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    
    plt.imsave(save_path, rgb)

def sobel_edges(x):
    """
    Compute Sobel edge map from a float tensor [B,C,H,W] or [B,1,H,W].
    and get edge magnitude [B,1,H,W].
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    x_gray = x.mean(dim=1, keepdim=True) if x.shape[1] != 1 else x
    edge_x = F.conv2d(x_gray, sobel_x, padding=1)
    edge_y = F.conv2d(x_gray, sobel_y, padding=1)
    return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)


def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)   #(1, 1, k, k)


def apply_random_blur(img, max_kernel_size=9):
    if img.ndim == 3:
        img = img.unsqueeze(0)   #[1,C,H,W]

    blur_type = random.choice(["gaussian", "motion"])

    if blur_type == "gaussian":
        ksize = random.choice([3, 5, 7, max_kernel_size])
        sigma = random.uniform(0.5, 2.5)
        ax    = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(img.device)
        kernel = kernel.repeat(img.shape[1], 1, 1, 1)
        blurry = F.conv2d(img, kernel, padding=ksize // 2, groups=img.shape[1])
    else:
        ksize  = random.choice([3, 5, 7, max_kernel_size])
        kernel = torch.zeros((ksize, ksize))
        if random.random() < 0.5:
            kernel[ksize // 2, :] = 1.0   #horizontal
        else:
            kernel[:, ksize // 2] = 1.0   #vertical
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(img.device)
        kernel = kernel.repeat(img.shape[1], 1, 1, 1)
        blurry = F.conv2d(img, kernel, padding=ksize // 2, groups=img.shape[1])

    return blurry.squeeze(0)


def match_size(a, b):
    """Resize tensor a to match tensor b's spatial size (H, W)."""
    if a.shape[-2:] != b.shape[-2:]:
        a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
    return a


def _to_uint8(img):
    """
    Convert a [C, H, W] float tensor to a [H, W, C] uint8 numpy array.two input ranges:
      [-1, 1]  (tanh output)  -> (x + 1) / 2 * 255
      [ 0, 1]  (sigmoid output) -> x * 255

    """
    img = img.detach().float()
    img_min = img.min().item()

    if img_min < -0.01:
        #Tanh range [-1, 1]
        img = (img + 1.0) / 2.0

    img = (img.clamp(0.0, 1.0) * 255.0).round()
    return img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def save_tensor_as_image(tensor, path):
    """
    Save a single [C, H, W] generator output tensor as a PNG.
    FIX: replaced deprecated .byte() with .to(torch.uint8) via _to_uint8.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = _to_uint8(tensor.squeeze(0) if tensor.ndim == 4 else tensor)
    imageio.imwrite(path, img)


def deblur_with_tiling(model, img, patch_size=256, overlap=32, device="cuda"):
    """
    Deblur a full image by processing overlapping patches and blending with
    Gaussian weights to avoid visible grid artifacts at patch boundaries and
    returns deblurred image tensor [1, C, H, W].
    """
    _, C, H, W = img.shape
    stride     = patch_size - overlap
    out_img    = torch.zeros(1, C, H, W, device=device)
    weight_map = torch.zeros(1, 1, H, W, device=device)

    ax     = torch.linspace(-1, 1, patch_size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    sigma  = 0.5
    gauss  = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gauss  = (gauss / gauss.max()).unsqueeze(0).unsqueeze(0)   #(1,1,P,P)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end   = min(y + patch_size, H)
            x_end   = min(x + patch_size, W)
            y_start = max(y_end - patch_size, 0)
            x_start = max(x_end - patch_size, 0)

            patch = img[:, :, y_start:y_end, x_start:x_end].to(device)
            ph, pw = patch.shape[2], patch.shape[3]
            w = gauss[:, :, :ph, :pw]

            with torch.no_grad():
                out_patch = model(patch)["out"]

            out_img[:, :, y_start:y_end, x_start:x_end]    += out_patch * w
            weight_map[:, :, y_start:y_end, x_start:x_end] += w

    return out_img / weight_map.clamp(min=1e-6)

#discriminator loss weights 
def get_disc_weights(epoch):
    if epoch < 100:
        return {"patch": 0.7, "freq": 0.3, "featmatch": 0.2}
    elif epoch < 300:
        return {"patch": 0.6, "freq": 0.4, "featmatch": 0.2}
    elif epoch < 700:
        return {"patch": 0.5, "freq": 0.5, "featmatch": 0.3}
    else:
        return {"patch": 0.4, "freq": 0.6, "featmatch": 0.3}

class MultiSaver():
    def __init__(self, result_dir=None):
        self.queue      = None
        self.process    = None
        self.result_dir = result_dir

    def begin_background(self):
        self.queue = mp.Queue()

        def _worker(queue):
            while True:
                if queue.empty():
                    continue
                item = queue.get()
                if item is None:
                    return
                img, name = item
                if name:
                    try:
                        basename, ext = os.path.splitext(name)
                        if ext != '.png':
                            name = f'{basename}.png'
                        imageio.imwrite(name, img)
                    except Exception as e:
                        print(f"[MultiSaver] Failed to write {name}: {e}")

        cpu_count = min(8, max(1, mp.cpu_count() - 1))
        self.process = [
            mp.Process(target=_worker, args=(self.queue,), daemon=False)
            for _ in range(cpu_count)
        ]
        for p in self.process:
            p.start()

    def end_background(self):
        if self.queue is None:
            return
        for _ in self.process:
            self.queue.put(None)

    def join_background(self):
        if self.queue is None:
            return
        while not self.queue.empty():
            time.sleep(0.5)
        for p in self.process:
            p.join()
        self.queue = None

    def save_image(self, output, save_names, result_dir=None):
        result_dir = result_dir if result_dir is not None else self.result_dir
        if result_dir is None:
            raise ValueError('No result_dir specified for MultiSaver.')

        if self.queue is None:
            self.begin_background()

        #4D NCHW
        if output.ndim == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.ndim == 3:
            output = output.unsqueeze(0)

        for output_img, save_name in zip(output, save_names):
            img_np = _to_uint8(output_img)

            full_path = os.path.join(result_dir, save_name)
            save_dir = os.path.dirname(full_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            self.queue.put((img_np, full_path))


def interact(local=None):
    """Interactive console with autocomplete. Usage: interact(locals())"""
    if local is None:
        local = dict(globals(), **locals())
    try:
        readline.set_completer(rlcompleter.Completer(local).complete)
    except NameError:
        pass
    code.interact(local=local)


def set_trace(local=None):
    """Debugging with pdb."""
    if local is None:
        local = dict(globals(), **locals())
    try:
        pdb.Pdb.complete = rlcompleter.Completer(local).complete
    except NameError:
        pass
    pdb.set_trace()


def visualize_attention(attn_tensor, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    attn_map = attn_tensor.squeeze().detach().cpu().numpy()
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def log_attention_weights(alpha_local, alpha_global):
    print(f"Alpha Local: {alpha_local.item():.4f}, Alpha Global: {alpha_global.item():.4f}")

class Timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret      = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() == 'true':
        return True
    if val.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def int2str(val):
    if isinstance(val, int):
        return str(val)
    if isinstance(val, str):
        return val
    raise argparse.ArgumentTypeError('Number value expected.')


class Map(dict):
    """Dict with attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]

    def toDict(self):
        return self.__dict__