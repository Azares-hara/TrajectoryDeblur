from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
class WarmMultiStepLR(_LRScheduler):
    """
    MultiStepLR with linear warm-up.Warm-up: linearly increase LR from base_lr/scale to base_lr overwarmup_epochs steps
    After warm-up,decay LR by gamma at each milestone epoch.
    Args:
        optimizer:      wrapped optimizer
        milestones:     list of epoch indices at which to decay LR
        gamma:          decay factor at each milestone (default 0.5)
        last_epoch:     last epoch index for resuming (-1 = fresh start)
        scale:          warmup start factor -- LR begins at base_lr/scale
        warmup_epochs:  number of epochs to ramp up over
    """

    def __init__(self, optimizer, milestones, gamma=0.5,
                 last_epoch=-1, scale=10, warmup_epochs=5):
        self.milestones    = sorted(milestones)
        self.gamma         = gamma
        self.scale         = scale
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_epochs:
                progress = (self.last_epoch + 1) / self.warmup_epochs
                lr = base_lr * (1.0 / self.scale + progress * (1.0 - 1.0 / self.scale))
            else:
                lr = base_lr * (self.gamma ** bisect_right(self.milestones, self.last_epoch))
            lrs.append(lr)
        return lrs