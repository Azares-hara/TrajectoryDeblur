
import torch.serialization
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
import os
import torch
import warnings
import traceback
from utils import interact
from option import args, setup, cleanup
from model2 import Model
from lossfunc import Loss
from optimizer import Optimizer
from train import Trainer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main_worker(rank, args):
    args.rank = rank

    if args.device_type == "cuda" and torch.cuda.is_available():
        args.device = f"cuda:{getattr(args, 'device_index', 0)}"
    else:
        args.device = "cpu"
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Dataset path not found: {args.data_root}")

    args = setup(args)

    print("Training with the following configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    model = Model(args)
    model.parallelize()
    print(f"Instantiating model with args.model = {args.model}")
    pretrained_path = '/kaggle/working/model-100-pretrained.pt'
    if os.path.exists(pretrained_path):
        print(f"[INFO] Loading pretrained weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=args.device)
        model.load_state_dict(ckpt, strict=False)
        print("[INFO] Loaded pretrained generator weights from epoch 100")
    else:
        print(f"[WARNING] No pretrained weights found at {pretrained_path}")
    optimizer = Optimizer(args, model)
    criterion = Loss(args, model=model, optimizer=optimizer)
    trainer   = Trainer(args, model, criterion, optimizer)
    
  
    if getattr(args, 'reset_discriminator', False):
        from model2.discriminator import MultiScaleDiscriminator
        from model2.model import init_weights
        trainer.discriminator = MultiScaleDiscriminator(
            n_feats=args.n_feats
        ).to(torch.device(args.device))
        trainer.discriminator.apply(init_weights)
        trainer.optimizer_D = torch.optim.Adam(
            trainer.discriminator.parameters(),
            lr=args.lr_D, betas=(0.5, 0.999)
        )
        print("[INFO] Discriminator weights reset.")
    
    if args.stay:
        interact(local=locals())
        return

    if args.demo:
        trainer.evaluate(epoch=args.start_epoch, mode='demo')
        return

    for epoch in range(1, args.start_epoch):
        if args.do_validate and epoch % args.validate_every == 0:
            trainer.fill_evaluation(epoch, 'val')
        if args.do_test and epoch % args.test_every == 0:
            trainer.evaluate(epoch, 'test')

    #main training loop
    if args.do_train:
        trainer.train(start_epoch=args.start_epoch, num_epochs=args.end_epoch + 1)
        trainer.finish()
    if args.do_validate and args.rank == 0:
        last_epoch = trainer.epoch
        print(f"[INFO] Running final validation at epoch {last_epoch}")
        trainer.validate(last_epoch)

    if args.do_test and args.rank == 0:
        last_epoch = trainer.epoch
        print(f"[INFO] Running final test at epoch {last_epoch}")
        trainer.test(last_epoch)

    trainer.imsaver.join_background()
    cleanup(args)


def main():
    n_gpus = getattr(args, 'n_GPUs', 1)
    distributed = getattr(args, 'distributed', False)

    if distributed and n_gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(args,), nprocs=n_gpus, join=True)
    else:
        #single gpu or cpu 
        try:
            args.rank = 0
            main_worker(0, args)
        except Exception as e:
            print(f"[ERROR] Job failed: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()