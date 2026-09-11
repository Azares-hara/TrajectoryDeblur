import random
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import pyramid_gaussian
import torch
import skimage
from packaging import version
import torch.nn.functional as F

USE_CHANNEL_AXIS = version.parse(skimage.__version__) >= version.parse("0.19")


def _apply(func, x):
    if isinstance(x, (list, tuple)):
        return [_apply(func, x_i) for x_i in x]
    elif isinstance(x, dict):
        return {key: _apply(func, value) for key, value in x.items()}
    else:
        return func(x)


def crop(*args, ps=256):
    def _get_shape(*args):
        if isinstance(args[0], (list, tuple)):
            return _get_shape(args[0][0])
        elif isinstance(args[0], dict):
            return _get_shape(list(args[0].values())[0])
        else:
            return args[0].shape

    h, w, _ = _get_shape(args)
    py = random.randrange(0, h - ps + 1)
    px = random.randrange(0, w - ps + 1)

    def _crop(img):
        if img.ndim == 2:
            return img[py:py+ps, px:px+ps, np.newaxis]
        else:
            return img[py:py+ps, px:px+ps, :]

    return _apply(_crop, args)


def add_noise(*args, sigma_sigma=2, rgb_range=255):
    if len(args) == 1:
        args = args[0]

    sigma = np.random.normal() * sigma_sigma * rgb_range / 255

    def _add_noise(img):
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return (img + noise).clip(0, rgb_range)

    return _apply(_add_noise, args)


def augment(*args, hflip=True, rot=True, shuffle=True,
            change_saturation=True, rgb_range=255):
    choices = (False, True)
    hflip  = hflip and random.choice(choices)
    vflip  = rot   and random.choice(choices)
    rot90  = rot   and random.choice(choices)

    if shuffle:
        rgb_order = list(range(3))
        random.shuffle(rgb_order)
        shuffle = rgb_order != list(range(3))

    if change_saturation:
        amp_factor = np.random.uniform(0.5, 1.5)

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        if shuffle and img.ndim > 2 and img.shape[-1] == 3:
            img = img[..., rgb_order]
        if change_saturation:
            img_norm = img / rgb_range
            hsv_img  = rgb2hsv(img_norm)
            hsv_img[..., 1] = np.clip(hsv_img[..., 1] * amp_factor, 0, 1)
            img = (hsv2rgb(hsv_img).clip(0, 1) * rgb_range)
        return img.astype(np.float32)
    return _apply(_augment, args)

def pad(img, divisor=4, pad_width=None, negative=False):
    def _pad_numpy(img, divisor, pad_width, negative):
        if pad_width is None:
            h, w, _ = img.shape
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        return np.pad(img, pad_width, mode='edge'), pad_width

    def _pad_tensor(img, divisor, pad_width, negative):
        n, c, h, w = img.shape
        if pad_width is None:
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = (0, pad_w, 0, pad_h)
        else:
            try:
                pad_h = pad_width[0][1]
                pad_w = pad_width[1][1]
                pad_width = (0, pad_w, 0, pad_h)
            except Exception:
                pass
            if negative:
                pad_width = [-val for val in pad_width]
        return torch.nn.functional.pad(img, pad_width, 'reflect'), pad_width

    if isinstance(img, np.ndarray):
        return _pad_numpy(img, divisor, pad_width, negative)
    else:
        return _pad_tensor(img, divisor, pad_width, negative)


def generate_pyramid(*args, n_scales):
    def _generate_pyramid(img):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if USE_CHANNEL_AXIS:
            return list(pyramid_gaussian(img, n_scales - 1, channel_axis=-1))
        else:
            return list(pyramid_gaussian(img, n_scales - 1, multichannel=True))

    return [_generate_pyramid(arg) for arg in args]


def np2tensor(*imgs):
    """
    Convert numpy images (HWC, [0,255]) to PyTorch tensors (CHW, [-1,1]).
    """
    def _convert(img):
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)
        tensor = tensor.div(255.0)          # [0, 255] ? [0, 1]
        tensor = tensor.mul(2.0).sub(1.0)   # [0,   1] ? [-1, 1]
        return tensor

    tensors = []
    for img in imgs:
        if isinstance(img, list):
            for sub in img:
                if isinstance(sub, np.ndarray):
                    tensors.append(_convert(sub))
                else:
                    raise TypeError(f"Expected numpy.ndarray, got {type(sub)}")
        elif isinstance(img, np.ndarray):
            tensors.append(_convert(img))
        else:
            raise TypeError(f"Expected numpy.ndarray, got {type(img)}")
    return tensors[0] if len(tensors) == 1 else tensors


def pad_to_max(tensors):
    tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in tensors]
    max_h = max(t.shape[2] for t in tensors)
    max_w = max(t.shape[3] for t in tensors)
    padded = []
    for t in tensors:
        pad_h = max_h - t.shape[2]
        pad_w = max_w - t.shape[3]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded.append(F.pad(t, (pad_left, pad_right, pad_top, pad_bottom)))
    return padded


def to(input, target, device=None, dtype=torch.float32):
    def _process_input(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype, non_blocking=True)
        elif isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], list):
                S = len(x[0])
                per_scale = []
                for s in range(S):
                    tensors = [item[s] for item in x]
                    tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in tensors]
                    tensors = pad_to_max(tensors)
                    per_scale_tensor = torch.cat(tensors, dim=0).to(
                        device=device, dtype=dtype, non_blocking=True)
                    per_scale.append(per_scale_tensor)
                return per_scale
            else:
                tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in x]
                tensors = pad_to_max(tensors)
                return torch.cat(tensors, dim=0).to(
                    device=device, dtype=dtype, non_blocking=True)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def _process_target(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            return x.to(device=device, dtype=dtype, non_blocking=True)
        elif isinstance(x, list):
            tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in x]
            stacked = torch.cat(pad_to_max(tensors), dim=0)
            return stacked.to(device=device, dtype=dtype, non_blocking=True)
        else:
            raise TypeError(f"Unsupported target type: {type(x)}")

    return _process_input(input), _process_target(target)
