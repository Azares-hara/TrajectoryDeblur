import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import imageio.v2 as imageio
from dataset import common
from utils import apply_random_blur

class Dataset(data.Dataset):
    """Base dataset class for blur/sharp image pairs/unpaired blur images."""
    def __init__(self, args, mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.modes = ()
        self.set_modes()
        self._check_mode()
        self.set_keys()

        if self.mode == 'demo':
            self.subset_root = args.demo_input_dir
        else:
            self.subset_root = args.data_root

    def set_keys(self):
        self.blur_key = 'blur'
        self.sharp_key = 'sharp'
        if hasattr(self.args, "blur_type"):
            if self.args.blur_type == "synthetic":
                self.blur_key = "synthetic_blur"
            elif self.args.blur_type == "gamma":
                self.blur_key = "blur_gamma"
        self.non_blur_keys  = []
        self.non_sharp_keys = []

    def set_modes(self):
        self.modes = ('train', 'val', 'test', 'demo')

    def _check_mode(self):
        if self.mode not in self.modes:
            raise NotImplementedError(f'mode error: not implemented for {self.mode}')

    def _scan(self,root=None):
        if root is None:
            root = self.subset_root

        if self.blur_key in self.non_blur_keys: self.non_blur_keys.remove(self.blur_key)
        if self.sharp_key in self.non_sharp_keys: self.non_sharp_keys.remove(self.sharp_key)

        def _key_check(path, true_key, false_keys):
            path = os.path.join(path, '')
            if path.find(true_key) < 0:
                return False
            return all(path.find(fk) < 0 for fk in false_keys)

        def _get_list(root, true_key, false_keys):
            data_list = []
            for sub, dirs, files in os.walk(root):
                if not dirs:
                    if _key_check(sub, true_key, false_keys):
                        data_list += [os.path.join(sub, f) for f in files]
            data_list.sort()
            return data_list

        self.blur_key = os.path.join(self.blur_key,  '')
        self.sharp_key = os.path.join(self.sharp_key, '')
        self.non_blur_keys = [os.path.join(k, '') for k in self.non_blur_keys]
        self.non_sharp_keys = [os.path.join(k, '') for k in self.non_sharp_keys]

        self.blur_list = _get_list(root, self.blur_key,  self.non_blur_keys)
        self.sharp_list = _get_list(root, self.sharp_key, self.non_sharp_keys)

    def _load_image(self, path):
        """load image as uint8 HWC numpy array, always RGB (3 channels)."""
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1) 
        elif img.shape[2] == 4:
            img = img[:, :, :3]                   
        return img.astype(np.float32)

    def __getitem__(self, idx):
        blur  = self._load_image(self.blur_list[idx])
        sharp = None
        if idx < len(self.sharp_list):
            try:
                sharp = self._load_image(self.sharp_list[idx])
            except Exception:
                sharp = None

        relpath = os.path.relpath(self.blur_list[idx], self.subset_root)
        patch_size = getattr(self.args, 'patch_size', 256)
        if self.mode == 'train' and patch_size > 0:
            h, w = blur.shape[:2]
            if h >= patch_size and w >= patch_size:
                if sharp is not None:
                    blur, sharp = common.crop(blur, sharp, ps=patch_size)
                else:
                    blur, = common.crop(blur, ps=patch_size)
            blur, sharp = (
                common.augment(blur, sharp) if sharp is not None
                else (common.augment(blur)[0], None)
            )

        
        if getattr(self.args, "gaussian_pyramid", False):
            n_scales = getattr(self.args, "n_scales", 1)
            if sharp is not None:
                blur_pyr, sharp_pyr = common.generate_pyramid(blur, sharp, n_scales=n_scales)
                blur  = common.np2tensor(*blur_pyr)
                sharp = common.np2tensor(*sharp_pyr)
            else:
                blur_pyr, = common.generate_pyramid(blur, n_scales=n_scales)
                blur  = common.np2tensor(*blur_pyr)
        else:
            blur = common.np2tensor(blur)
            sharp = common.np2tensor(sharp) if sharp is not None else None

        return {
            "blur":blur,
            "sharp":sharp,
            "blur_path":relpath,
            "idx":idx,
        }

    def __len__(self):
        return len(self.blur_list)


class BlurDataset(torch.utils.data.Dataset):
    """
    Simple paired dataset that loads blur/sharp from two directories.
    """

    def __init__(self, blur_dir, sharp_dir, patch_size=256, augment=True):
        self.blur_paths = sorted(glob.glob(os.path.join(blur_dir,"*.png")))
        self.sharp_paths = sorted(glob.glob(os.path.join(sharp_dir,"*.png")))
        self.patch_size = patch_size
        self.do_augment = augment
        assert len(self.blur_paths) == len(self.sharp_paths), (
            f"Blur/sharp count mismatch: {len(self.blur_paths)} vs {len(self.sharp_paths)}"
        )

    def _load(self, path):
        img = imageio.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def __getitem__(self, idx):
        blur  = self._load(self.blur_paths[idx])
        sharp = self._load(self.sharp_paths[idx])

        if self.patch_size > 0:
            blur, sharp = common.crop(blur, sharp, ps=self.patch_size)
        if self.do_augment:
            blur, sharp = common.augment(blur, sharp)

        blur  = common.np2tensor(blur)
        sharp = common.np2tensor(sharp)
        return blur, sharp

    def __len__(self):
        return len(self.sharp_paths)