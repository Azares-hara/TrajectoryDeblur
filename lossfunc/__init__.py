import os
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import interact
from .frequencyloss import FrequencyLoss
from .adversarial import Adversarial
from lossfunc.metric import PSNR, SSIM, LPIPSMetric, NIQEMetric, BRISQUEMetric
from lossfunc.lpipsregistry import get_lpips

__all__ = [
    "FrequencyLoss",
    "Adversarial",
    "PSNR",
    "SSIM",
    "LPIPSMetric",
    "NIQEMetric",
    "BRISQUEMetric",
    "NeighborLoss",
    "get_default_device",
]


def get_default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class NeighborLoss(nn.Module):
    """
    Penalises differences in local pixel-neighbour gradients between output and
    and target. Encourages spatial smoothness in non-blurry regions.
    """
    def forward(self, output, target):
        #vertical neighbours
        diff_out_v = output[:, :, :-1, :] - output[:, :, 1:, :]
        diff_tgt_v = target[:, :, :-1, :] - target[:, :, 1:, :]
        #horizontal neighbours
        diff_out_h = output[:, :, :, :-1] - output[:, :, :, 1:]
        diff_tgt_h = target[:, :, :, :-1] - target[:, :, :, 1:]
        return (
            F.l1_loss(diff_out_v, diff_tgt_v) +
            F.l1_loss(diff_out_h, diff_tgt_h)
        )


class Loss(torch.nn.modules.loss._Loss):
    def __init__(self, args, epoch=None, model=None, optimizer=None):
        super().__init__()

        self.args        = args
        self.rgb_range   = args.rgb_range
        self.device_type = args.device_type
        self.synchronized = False

        self.epoch    = args.start_epoch if epoch is None else epoch
        self.save_dir = args.save_dir
        self.save_name = os.path.join(self.save_dir, 'loss.pt')

        self.validating = False
        self.testing    = False
        self.mode       = 'train'
        self.modes      = ('train', 'val', 'test')

        self.loss       = nn.ModuleDict()
        self.loss_types = []
        self.weight     = {}
        self.loss_stat  = {mode: {} for mode in self.modes}

        for weighted_loss in args.loss.split('+'):
            if '*' in weighted_loss:
                w, l = weighted_loss.split('*')
            else:
                w, l = 1.0, weighted_loss
            l = l.strip()

            try:
                if l in ('ABS', 'L1'):
                    loss_type = 'L1'
                    func = nn.L1Loss()
                elif l in ('MSE', 'L2'):
                    loss_type = 'L2'
                    func = nn.MSELoss()
                elif l in ('ADV', 'GAN'):
                    loss_type = 'ADV'
                    func = Adversarial(args, model, optimizer)
                elif l == 'NeighborLoss':
                    loss_type = 'NeighborLoss'
                    func = NeighborLoss()
                elif l == 'LPIPS':
                    loss_type = 'LPIPS'
                    func = get_lpips(net="squeeze", device=args.device, use_half=False)
                elif l == 'FrequencyLoss':
                    loss_type = 'FrequencyLoss'
                    func = FrequencyLoss(
                        weight=getattr(args, 'lambda_freq', 0.05),
                        use_phase=True,
                        multi_scale=True,
                        scales=(2, 4),
                        return_components=False,
                    )
                else:
                    raise ValueError(f"Unknown loss type '{l}'")

            except (ModuleNotFoundError, AttributeError, ValueError) as e:
                raise ValueError(f"[ERROR] Could not construct loss '{l}': {e}")

            self.loss_types.append(loss_type)
            self.loss[loss_type] = func
            self.weight[loss_type] = float(w)

        print(f'Loss function: {args.loss}')

        #metrics
        self.do_measure  = args.metric.lower() != 'none'
        self.metric      = nn.ModuleDict()
        self.metric_types = []
        self.metric_stat = {mode: {} for mode in self.modes}

        if self.do_measure:
            for metric_type in args.metric.split(','):
                metric_type = metric_type.strip().upper()
                if metric_type == 'PSNR':
                    metric_func = PSNR(device=args.device)
                elif metric_type == 'SSIM':
                    metric_func = SSIM(device_type=args.device_type)
                elif metric_type == 'LPIPS':
                    metric_func = get_lpips(net="squeeze", device=args.device, use_half=False)
                elif metric_type == 'NIQE':
                    metric_func = NIQEMetric()
                elif metric_type == 'BRISQUE':
                    metric_func = BRISQUEMetric()
                else:
                    raise NotImplementedError(f"Metric '{metric_type}' not implemented.")
                self.metric_types.append(metric_type)
                self.metric[metric_type] = metric_func

        print(f'Metrics: {args.metric}')

        if args.start_epoch != 1:
            self.load(args.start_epoch - 1)

        for mode in self.modes:
            for loss_type in self.loss:
                self.loss_stat[mode].setdefault(loss_type, {})
            self.loss_stat[mode].setdefault('Total', {})
            if self.do_measure:
                for metric_type in self.metric:
                    self.metric_stat[mode].setdefault(metric_type, {})

        self.count   = 0
        self.count_m = 0
        self.to(args.device, dtype=args.dtype)


    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.validating = False
            self.testing    = False
            self.mode       = 'train'
        else:
            self.validating = False
            self.testing    = True
            self.mode       = 'test'

    def validate(self):
        super().eval()
        self.validating = True
        self.testing    = False
        self.mode       = 'val'

    def test(self):
        super().eval()
        self.validating = False
        self.testing    = True
        self.mode       = 'test'

    def forward(self, input, target):
        self.synchronized = False
        loss = 0

        if self.count == 0:
            for loss_type in self.loss_types:
                self.loss_stat[self.mode][loss_type][self.epoch] = 0
            self.loss_stat[self.mode]['Total'][self.epoch] = 0

        count = input[0].shape[0] if isinstance(input, list) else input.shape[0]
        isnan = False

        for loss_type in self.loss_types:
            if loss_type == 'ADV':
                fake = input[0]  if isinstance(input,  list) else input
                real = target[0] if isinstance(target, list) else target
                adv_result = self.loss[loss_type](fake, real, self.training)
                _loss = (adv_result[0] if isinstance(adv_result, tuple) else adv_result)
                _loss = _loss * self.weight[loss_type]
            else:
                _loss = self._ms_forward(input, target, self.loss[loss_type])
                _loss = _loss * self.weight[loss_type]

            if torch.isnan(_loss):
                isnan = True
            else:
                self.loss_stat[self.mode][loss_type][self.epoch] += _loss.item() * count
                self.loss_stat[self.mode]['Total'][self.epoch] += _loss.item() * count
            loss += _loss

        if not isnan:
            self.count += count

        if not self.training and self.do_measure:
            self.measure(input, target)

        return loss

    def _ms_forward(self, input, target, func):
        if isinstance(input, (list, tuple)):
            if isinstance(target, (list, tuple)):
                if len(target) != len(input):
                    raise ValueError("Input/target scale count mismatch.")
                return sum(func(i, t) for i, t in zip(input, target))
            return sum(func(i, target) for i in input)
        if isinstance(input, dict):
            if isinstance(target, dict):
                return sum(func(input[k], target[k]) for k in input)
            return sum(func(input[k], target) for k in input)
        return func(input, target)

    def measure(self, input, target):
        if isinstance(input, (list, tuple)):
            return self.measure(
                input[0],
                target[0] if isinstance(target, (list, tuple)) else target
            )
        if isinstance(input, dict):
            k = next(iter(input))
            return self.measure(
                input[k],
                target[k] if isinstance(target, dict) else target
            )

        if self.count_m == 0:
            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][self.epoch] = 0

        count = input.shape[0]
        input = input.clamp(0, self.rgb_range)
        if self.rgb_range == 255:
            input = input.round()

        for metric_type in self.metric_stat[self.mode]:
            if metric_type == 'LPIPS':
                inp_n = (input  / self.rgb_range) * 2 - 1
                tgt_n = (target / self.rgb_range) * 2 - 1
                val = self.metric[metric_type](inp_n, tgt_n)
            else:
                val = self.metric[metric_type](input, target)
            self.metric_stat[self.mode][metric_type][self.epoch] += val.item() * count

        self.count_m += count


    def normalize(self):
        if self.args.distributed:
            dist.barrier()
            if not self.synchronized:
                self.all_reduce()

        if self.count > 0:
            for loss_type in self.loss_stat[self.mode]:
                self.loss_stat[self.mode][loss_type][self.epoch] /= self.count
            self.count = 0

        if self.count_m > 0:
            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][self.epoch] /= self.count_m
            self.count_m = 0

    def all_reduce(self, epoch=None):
        if epoch is None:
            epoch = self.epoch

        def _reduce(value):
            t = torch.tensor([value], device=self.args.device, dtype=self.args.dtype)
            dist.all_reduce(t, dist.ReduceOp.SUM, async_op=False)
            return t.item()

        dist.barrier()
        if self.count > 0:
            self.count = _reduce(self.count)
            for loss_type in self.loss_stat[self.mode]:
                self.loss_stat[self.mode][loss_type][epoch] = _reduce(
                    self.loss_stat[self.mode][loss_type][epoch]
                )
        if self.count_m > 0:
            self.count_m = _reduce(self.count_m)
            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][epoch] = _reduce(
                    self.metric_stat[self.mode][metric_type][epoch]
                )
        self.synchronized = True


    def get_last_loss(self):
        return self.loss_stat[self.mode]['Total'].get(self.epoch, 0.0)

    def get_loss_desc(self):
        prefix = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.get(self.mode, self.mode)
        loss = self.loss_stat[self.mode]['Total'].get(self.epoch, 0.0)
        desc = f'{prefix} Loss: {loss:.4f}'
        if self.mode in ('val', 'test'):
            desc += self.get_metric_desc()
        return desc

    def get_metric_desc(self):
        desc = ''
        with torch.no_grad():
            for metric_type, metric_record in self.metric_stat[self.mode].items():
                measured = metric_record.get(self.epoch, 0.0)
                if metric_type == 'PSNR':
                    desc += f' PSNR: {measured:2.2f}'
                elif metric_type == 'SSIM':
                    desc += f' SSIM: {measured:1.4f}'
                elif metric_type == 'LPIPS':
                    desc += f' LPIPS: {measured:1.4f}'
                else:
                    desc += f' {metric_type}: {measured:2.4f}'
        return desc

    def step(self, plot_name=None):
        self.normalize()
        self.plot(plot_name)
        if not self.training and self.do_measure:
            self.plot_metric()

    def save(self):
        torch.save(
            {'loss_stat': self.loss_stat, 'metric_stat': self.metric_stat},
            self.save_name
        )

    def load(self, epoch=None):
        print(f"Loading loss record from {self.save_name}")
        if os.path.exists(self.save_name):
            try:
                state = torch.load(self.save_name, map_location=self.args.device, weights_only=True)
            except TypeError:
                print("[WARNING] Fallback to full object loading.")
                state = torch.load(self.save_name, map_location=self.args.device)
            self.loss_stat   = state.get('loss_stat',   self.loss_stat)
            self.metric_stat = state.get('metric_stat', self.metric_stat)
        else:
            print(f"No loss record found at {self.save_name}.")
        if epoch is not None:
            self.epoch = epoch

    def plot(self, plot_name=None, metric=False):
        self.plot_loss(plot_name)
        if metric:
            self.plot_metric(plot_name)

    def plot_loss(self, plot_name=None):
        if plot_name is None:
            plot_name = os.path.join(self.save_dir, f"{self.mode}_loss.pdf")
        style_map = {
            'L1':    {'color': 'red'},
            'ADV':   {'color': 'gold'},
            'Total': {'color': 'indigo'},
        }
        fig = plt.figure()
        plt.title(f"{self.mode} loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(True, linestyle=':')
        for loss_type, record in self.loss_stat[self.mode].items():
            axis  = sorted(e for e in record if e <= self.epoch)
            value = [record[e] for e in axis]
            plt.plot(axis, value, label=loss_type, **style_map.get(loss_type, {}))
        if self.epoch > 0:
            plt.xlim(0, self.epoch)
        plt.legend()
        plt.savefig(plot_name)
        plt.close(fig)

    def plot_metric(self, plot_name=None):
        if plot_name is None:
            plot_name = os.path.join(self.save_dir, f"{self.mode}_metric.pdf")
        fig, ax1 = plt.subplots()
        plt.title(f"{self.mode} metrics")
        plt.grid(True, linestyle=':')
        ax1.set_xlabel('epochs')
        plots = []
        for i, (metric_type, record) in enumerate(self.metric_stat[self.mode].items()):
            axis  = sorted(e for e in record if e <= self.epoch)
            value = [record[e] for e in axis]
            ax    = ax1.twinx() if metric_type == 'SSIM' else ax1
            ax.set_ylabel(metric_type)
            plots += ax.plot(axis, value, label=metric_type, color=f'C{i}')
        if plots:
            plt.legend(plots, [p.get_label() for p in plots])
        if self.epoch > 0:
            plt.xlim(0, self.epoch)
        plt.savefig(plot_name)
        plt.close(fig)

    def sort(self):
        for mode in self.modes:
            for loss_type in self.loss_stat[mode]:
                self.loss_stat[mode][loss_type] = dict(
                    sorted(self.loss_stat[mode][loss_type].items())
                )
            for metric_type in self.metric_stat[mode]:
                self.metric_stat[mode][metric_type] = dict(
                    sorted(self.metric_stat[mode][metric_type].items())
                )
        return self