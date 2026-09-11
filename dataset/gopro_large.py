import os
import numpy as np
import torch
import imageio.v2 as imageio
from dataset.dataset import Dataset
from dataset import common


class GOPRO_Large(Dataset):
    def __init__(self, args, mode='train'):
        super().__init__(args, mode)
        try:
            _ = os.listdir(args.data_root)
        except Exception as e:
            print(f"[ERROR] Could not list scenes at {args.data_root}: {e}")
        self._scan()
        print(f"[INFO] blur_list: {len(self.blur_list)}, sharp_list: {len(self.sharp_list)}")
        if len(self.blur_list) != len(self.sharp_list):
            print(f"[WARN] MISMATCH: {len(self.blur_list)} blur vs {len(self.sharp_list)} sharp")

    def set_modes(self):
        self.modes = ('train', 'val', 'test', 'demo')

    def set_keys(self):
        super().set_keys()

    def _load_image(self, path):
        img = imageio.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def __getitem__(self, idx):
        blur = self._load_image(self.blur_list[idx])
        sharp = None
        if idx < len(self.sharp_list):
            try:
                sharp = self._load_image(self.sharp_list[idx])
            except Exception:
                sharp = None

        relpath = os.path.relpath(self.blur_list[idx], self.subset_root)

        if self.mode == 'train':
            patch_size = getattr(self.args, 'patch_size', 256)
        else:
            patch_size = getattr(self.args, 'val_patch_size', 0)

        if patch_size > 0:
            h, w = blur.shape[:2]
            if h >= patch_size and w >= patch_size:
                if sharp is not None:
                    blur, sharp = common.crop(blur, sharp, ps=patch_size)
                else:
                    blur, = common.crop(blur, ps=patch_size)
            if self.mode == 'train':
                if sharp is not None:
                    blur, sharp = common.augment(blur, sharp)
                else:
                    blur, = common.augment(blur)
        elif self.mode == 'train':
            if sharp is not None:
                blur, sharp = common.augment(blur, sharp)
            else:
                blur, = common.augment(blur)

        if sharp is None:
            print(f"[WARN] sharp is None for idx {idx}, blur: {self.blur_list[idx]}")
            sharp = np.zeros_like(blur)

        blur = common.np2tensor(blur)
        sharp = common.np2tensor(sharp)

        return {
            "blur": blur,
            "sharp": sharp,
            "blur_path": relpath,
            "idx": idx,
        }

    def __len__(self):
        return len(self.blur_list)


class SharpDataset(Dataset):
    """
    Loads only sharp images (no blur counterpart).
    Used for the unpaired sharp reference distribution in adversarial training.
    """

    def __init__(self, args, mode='train'):
        super().__init__(args, mode)
        self._scan_sharp_only()

    def _scan_sharp_only(self):
        root = self.subset_root
        sharp_key = 'sharp'
        self.sharp_list = []
        for sub, dirs, files in os.walk(root):
            if sub.endswith(sharp_key) or sharp_key in sub.split(os.sep):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.sharp_list.append(os.path.join(sub, f))
        self.sharp_list.sort()
        self.blur_list = []

    def _load_image(self, path):
        img = imageio.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def __getitem__(self, idx):
        sharp = self._load_image(self.sharp_list[idx])

        patch_size = getattr(self.args, 'patch_size', 256)
        if self.mode == 'train' and patch_size > 0:
            h, w = sharp.shape[:2]
            if h >= patch_size and w >= patch_size:
                sharp, = common.crop(sharp, ps=patch_size)
            sharp, = common.augment(sharp)

        sharp = common.np2tensor(sharp)

        relpath = os.path.relpath(self.sharp_list[idx], self.subset_root)
        blur_placeholder = torch.zeros_like(sharp)

        return {
            "blur": blur_placeholder,
            "sharp": sharp,
            "blur_path": relpath,
            "idx": idx,
        }

    def __len__(self):
        return len(self.sharp_list)