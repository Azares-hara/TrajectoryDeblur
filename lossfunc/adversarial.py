import torch
import torch.nn as nn

try:
    import torch.amp as amp        # PyTorch 2.x
except ImportError:
    from torch.cuda import amp     # PyTorch 1.x

class Adversarial(nn.Module):
    """
    Adversarial loss wrapper for MultiScaleDiscriminator using hinge loss.
    """
    def __init__(self, args, model, optimizer):
        super().__init__()
        self.args      = args
        self.optimizer = optimizer

        model_dict = model.model if hasattr(model, 'model') else {}
        self.discriminator = model_dict['D'] if 'D' in model_dict else None

    def discriminator_loss(self, fake: torch.Tensor, real: torch.Tensor,
                           scaler=None) -> dict:
        """
        Compute and apply discriminator hinge loss.
        Pass the trainer's AMP scaler so we use one shared scaler, not two.

        Returns a dict of scalar loss values for logging.
        """
        if self.discriminator is None:
            return {"D_total": 0.0}

        device = self.args.device
        fake   = fake.to(device).detach()
        real   = real.to(device)
        out_real = self.discriminator(real)
        out_fake = self.discriminator(fake)
        loss_patch = (
            torch.relu(1.0 - out_real["score_full"]).mean()    + torch.relu(1.0 + out_fake["score_full"]).mean() +
            torch.relu(1.0 - out_real["score_half"]).mean()    + torch.relu(1.0 + out_fake["score_half"]).mean() +
            torch.relu(1.0 - out_real["score_quarter"]).mean() + torch.relu(1.0 + out_fake["score_quarter"]).mean()
        )
        loss_freq = (
            torch.relu(1.0 - out_real["score_freq"]).mean() +
            torch.relu(1.0 + out_fake["score_freq"]).mean()
        )
        total_loss_d = loss_patch + loss_freq
        opt_d = getattr(self.optimizer, 'D', None)
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(total_loss_d).backward()
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=getattr(self.args, 'clip_grad', 5.0)
                )
                scaler.step(opt_d)
            else:
                total_loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=getattr(self.args, 'clip_grad', 5.0)
                )
                opt_d.step()

        return {
            "D_patch": loss_patch.item(),
            "D_freq":  loss_freq.item(),
            "D_total": total_loss_d.item(),
        }

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            return torch.tensor(0.0, device=self.args.device)

        fake   = fake.to(self.args.device)
        out    = self.discriminator(fake)

        loss_g = -(
            out["score_full"].mean() +
            out["score_half"].mean() +
            out["score_quarter"].mean() +
            out["score_freq"].mean()
        )
        return loss_g

    def forward(self, fake: torch.Tensor, real: torch.Tensor,
                training: bool = True, scaler=None):
        loss_d = {}
        if training:
            loss_d = self.discriminator_loss(fake, real, scaler=scaler)

        loss_g = self.generator_loss(fake)

        if training:
            return loss_g, loss_d
        else:
            return loss_g
