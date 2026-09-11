import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import os
from collections import Counter
from model2 import Model
from utils import Map
from optimizer.warm_multi_step_lr import WarmMultiStepLR


class Optimizer(object):
    def __init__(self, args, model):
        self.args = args
        self.save_dir = os.path.join(self.args.save_dir, 'optim')
        os.makedirs(self.save_dir, exist_ok=True)

        if isinstance(model, Model):
            model = model.model

        #choosing optimizer class
        kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if args.optimizer == 'SGD':
            optimizer_class = optim.SGD
            kwargs_optimizer['momentum'] = args.momentum
        elif args.optimizer == 'ADAM':
            optimizer_class = optim.Adam
            kwargs_optimizer['betas'] = args.betas
            kwargs_optimizer['eps'] = args.epsilon
        elif args.optimizer == 'RMSPROP':
            optimizer_class = optim.RMSprop
            kwargs_optimizer['eps'] = args.epsilon
        elif args.optimizer == 'ADAMW':
            optimizer_class = optim.AdamW
            kwargs_optimizer['betas'] = args.betas
            kwargs_optimizer['eps'] = args.epsilon
        else:
            raise NotImplementedError(f"Optimizer {args.optimizer} not supported.")

        if args.scheduler == 'cosine':
            scheduler_class = lrs.CosineAnnealingLR
            kwargs_scheduler = {'T_max': args.end_epoch, 'eta_min': 1e-6}
        elif args.scheduler == 'plateau':
            scheduler_class = lrs.ReduceLROnPlateau
            kwargs_scheduler = {
                'mode': 'min',
                'factor': args.gamma,
                'patience': 10,
                'verbose': True,
                'threshold': 0,
                'threshold_mode': 'abs',
                'cooldown': 10,
            }
        elif args.scheduler == 'warm_multistep':
            scheduler_class = WarmMultiStepLR
            kwargs_scheduler = {
                'milestones': args.milestones,
                'gamma': args.gamma,
                'warmup_epochs': getattr(args, "warmup_epochs", 5),
                'scale': getattr(args, "scale", 1.0),
            }
        else:
            raise NotImplementedError(f"Scheduler {args.scheduler} not supported.")

        self.kwargs_optimizer = kwargs_optimizer
        self.scheduler_class = scheduler_class
        self.kwargs_scheduler = kwargs_scheduler

        class _Optimizer(optimizer_class):
            def __init__(self, model, args, scheduler_class, kwargs_scheduler):
                trainable = filter(lambda x: x.requires_grad, model.parameters())
                super().__init__(trainable, **kwargs_optimizer)
                self.args = args
                self._register_scheduler(scheduler_class, kwargs_scheduler)

            def _register_scheduler(self, scheduler_class, kwargs_scheduler):
                if scheduler_class is lrs.CosineAnnealingLR:
                    self.scheduler = scheduler_class(self,
                                                     T_max=kwargs_scheduler.get('T_max'),
                                                     eta_min=kwargs_scheduler.get('eta_min'))
                elif scheduler_class is lrs.ReduceLROnPlateau:
                    self.scheduler = scheduler_class(self, **kwargs_scheduler)
                elif scheduler_class is WarmMultiStepLR:
                    self.scheduler = scheduler_class(self,
                                                     milestones=kwargs_scheduler.get('milestones'),
                                                     gamma=kwargs_scheduler.get('gamma'),
                                                     warmup_epochs=kwargs_scheduler.get('warmup_epochs'),
                                                     scale=kwargs_scheduler.get('scale'))
                else:
                    raise NotImplementedError(f"Scheduler {scheduler_class} not supported.")

            def schedule(self, metrics=None):
                if isinstance(self.scheduler, lrs.ReduceLROnPlateau):
                    self.scheduler.step(metrics)
                else:
                    self.scheduler.step()

            def get_last_epoch(self):
                return self.scheduler.last_epoch

            def get_lr(self):
                return self.param_groups[0]['lr']

            def get_last_lr(self):
                return self.scheduler.get_last_lr()[0]

            def state_dict(self):
                state_dict = super().state_dict()
                state_dict['scheduler'] = self.scheduler.state_dict()
                return state_dict

            def load_state_dict(self, state_dict, epoch=None):
                super().load_state_dict(state_dict)
                self.scheduler.load_state_dict(state_dict['scheduler'])

                reschedule = False
                if isinstance(self.scheduler, lrs.MultiStepLR):
                    if (self.args.milestones != list(self.scheduler.milestones) or
                        self.args.gamma != self.scheduler.gamma):
                        reschedule = True

                if reschedule:
                    if epoch is None:
                        epoch = self.scheduler.last_epoch if self.scheduler.last_epoch > 1 else self.args.start_epoch - 1
                    self.scheduler.milestones = Counter(self.args.milestones)
                    self.scheduler.gamma = self.args.gamma
                    for i, group in enumerate(self.param_groups):
                        self.param_groups[i]['lr'] = group['initial_lr']
                        multiplier = 1
                        for milestone in self.scheduler.milestones:
                            if epoch >= milestone:
                                multiplier *= self.scheduler.gamma
                        self.param_groups[i]['lr'] *= multiplier

        #optimizers 
        self.G = _Optimizer(model['G'], args, scheduler_class, kwargs_scheduler)
        for group in self.G.param_groups:
            group['lr'] = args.lr
        self.G.scheduler.base_lrs = [args.lr for _ in self.G.scheduler.base_lrs]
        print(f"Generator Optimizer LR={self.get_lr():.2e}")

        disc_kwargs = {'lr': args.lr_D, 'weight_decay': 0,
                       'betas': args.betas, 'eps': args.epsilon}

        if 'D' in model and model['D'] is not None:
            self.D = _Optimizer(model['D'], args, scheduler_class, kwargs_scheduler)
            for group in self.D.param_groups:
                group['lr'] = disc_kwargs['lr']
                group['weight_decay'] = disc_kwargs['weight_decay']
            self.D.scheduler.base_lrs = [args.lr_D for _ in self.D.scheduler.base_lrs]
        else:
            self.D = None

        if getattr(args, "resume_optimizer", True):
            self.load(args.load_epoch)
        else:
            print("Starting with fresh optimizer state")

    def zero_grad(self):
        self.G.zero_grad()
        if self.D is not None:
            self.D.zero_grad()

    def step(self):
        self.G.step()
        if self.D is not None:
            self.D.step()

    def schedule(self, metrics=None):
        self.G.schedule(metrics)
        if self.D is not None:
            self.D.schedule(metrics)

    def get_last_epoch(self):
        return self.G.get_last_epoch()

    def get_lr(self):
        return self.G.get_lr()

    def get_last_lr(self):
        return self.G.get_last_lr()

    def state_dict(self):
        state_dict = Map()
        state_dict.G = self.G.state_dict()
        if self.D is not None:
            state_dict.D = self.D.state_dict()
        return state_dict.toDict()

    def load_state_dict(self, state_dict, epoch=None):
        state_dict = Map(**state_dict)
        self.G.load_state_dict(state_dict.G, epoch)
        if self.D is not None and hasattr(state_dict, "D"):
            self.D.load_state_dict(state_dict.D, epoch)

    def _save_path(self, epoch=None):
        epoch = epoch if epoch is not None else self.get_last_epoch()
        return os.path.join(self.save_dir, f'optim-{epoch:d}.pt')

    def save(self, epoch=None):
        if epoch is None:
            epoch = self.G.scheduler.last_epoch
        torch.save(self.state_dict(), self._save_path(epoch))

    def load(self, epoch):
        if epoch > 0 and getattr(self.args, "resume_optimizer", True):
            print(f'Loading optimizer from {self._save_path(epoch)}')
            try:
                state = torch.load(self._save_path(epoch), map_location=self.args.device)
                self.load_state_dict(state, epoch=epoch)
            except ValueError as e:
                print(f"[WARNING] Optimizer state mismatch: {e}")
                print("Skipping optimizer state, resuming with fresh optimizer.")
        elif epoch > 0 and not getattr(self.args, "resume_optimizer", True):
            print("Skipping optimizer state load, starting fresh.")
        elif epoch == 0:
            pass
        else:
            raise NotImplementedError
