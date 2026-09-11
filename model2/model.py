import os
import re
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from model2.generator import TraUNetGenerator
from model2.discriminator import MultiScaleDiscriminator


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args     = args
        self.device   = args.device
        self.n_GPUs   = args.n_GPUs
        self.save_dir = os.path.join(args.save_dir, 'models')
        os.makedirs(self.save_dir, exist_ok=True)
        self._model = {}

        #generator
        G = TraUNetGenerator(
            args,
            in_channels=3,
            out_channels=3,
            base_channels=args.n_feats,
            use_sa=args.use_sa,
            out_activation="tanh",
        )
        G.apply(init_weights)
        self._model['G'] = G
        self.add_module('G', G)

        lambda_adv = getattr(args, 'lambda_adv', 0.0)
        if lambda_adv > 0:
            D = MultiScaleDiscriminator(n_feats=args.n_feats)
            D.apply(init_weights)
            self._model['D'] = D
            self.add_module('D', D)
        else:
            self._model['D'] = None

        self.model = self._model

    def forward(self, x, stage=3):
        out_dict = self.model['G'](x, stage=stage)
        out_dict["out"] = torch.nan_to_num(
            out_dict["out"], nan=0.0, posinf=1.0, neginf=-1.0
        )
        out_dict["blur_prob"] = torch.nan_to_num(
            out_dict["blur_prob"], nan=0.0, posinf=1.0, neginf=0.0
        )
        return out_dict

    def parallelize(self):
        if self.n_GPUs <= 1 or str(self.args.device) == 'cpu':
            print("[INFO] Skipping DataParallel: single GPU or CPU mode.")
            return

        Parallel      = DistributedDataParallel if self.args.distributed else DataParallel
        parallel_args = (
            {"device_ids": [self.args.rank], "output_device": self.args.rank}
            if self.args.distributed else
            {"device_ids": list(range(self.n_GPUs)), "output_device": self.args.rank}
        )

        for key, submodel in self._model.items():
            if submodel is not None:
                wrapped = Parallel(submodel, **parallel_args)
                self._model[key] = wrapped
                self.add_module(key, wrapped)
        self.model = self._model

    def _inner(self, key):
        """Unwrap DataParallel/DistributedDataParallel to get the base module."""
        submodel = self._model.get(key)
        if submodel is None:
            return None
        return submodel.module if isinstance(
            submodel, (DataParallel, DistributedDataParallel)
        ) else submodel

    def state_dict(self):
        sd = {}
        for key in self._model:
            inner = self._inner(key)
            if inner is not None:
                sd[key] = inner.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        for key in self._model:
            if key not in state_dict:
                continue
            inner = self._inner(key)
            if inner is None:
                continue
            missing, unexpected = inner.load_state_dict(state_dict[key], strict=strict)
            if missing:
                print(f"  [{key}] Missing keys:{missing}")
            if unexpected:
                print(f"  [{key}] Unexpected keys:{unexpected}")

    def _save_path(self, epoch):
        return os.path.join(self.save_dir, f"model-{epoch}.pt")

    def save(self, epoch):
        torch.save(self.state_dict(), self._save_path(epoch))

    def load(self, epoch=None, path=None, strict=False):
        if path:
            model_name = path
        elif isinstance(epoch, int):
            if epoch < 0:
                epoch = self.get_last_epoch()
            if epoch == 0:
                return
            model_name = self._save_path(epoch)
        else:
            raise ValueError("Provide either an epoch number or a model path.")

        print(f"Loading model from {model_name} (strict={strict})")
        if not os.path.isfile(model_name):
            raise FileNotFoundError(f"Model file not found: {model_name}")

        state_dict = torch.load(model_name, map_location=self.args.device)
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded checkpoint (strict={strict})")

    def synchronize(self):
        if not self.args.distributed:
            return
        dist.barrier()
        vector = parameters_to_vector(self.parameters())
        dist.broadcast(vector, src=0)
        if self.args.rank != 0:
            vector_to_parameters(vector, self.parameters())
        del vector

    def get_last_epoch(self):
        if not os.path.isdir(self.save_dir):
            return 0
        model_list = [
            f for f in os.listdir(self.save_dir)
            if re.search(r'\d+', f)
        ]
        if not model_list:
            return 0
        model_list.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
        return int(re.findall(r'\d+', model_list[-1])[0])

    def print(self):
        print(self._model)
