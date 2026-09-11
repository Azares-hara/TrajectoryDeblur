# pylint: disable=C0103, C0301
import torch.serialization
import argparse
import datetime
import os
import re
import shutil
import time
import sys

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils import interact
from utils import str2bool, int2str

import template
#torch.serialization.add_safe_globals([argparse.Namespace])

parser = argparse.ArgumentParser(description='Dynamic Scene Deblurring')
parser.add_argument('--lambda_inpaint', type=float, default=0.1,help='weight for blur-guided inpainting loss')
parser.add_argument('--lambda_prior', type=float, default=0.1,help='weight for sharpness prior loss')
parser.add_argument('--loss', type=str, default='L1',help='loss function(s) to use, e.g. L1+LPIPS+FrequencyLoss+NeighborLoss+ADV')
parser.add_argument('--lambda_blur', type=float, default=0.05,help='weight for self-supervised blur prediction loss')
parser.add_argument('--synthetic_sharp_paths', type=str, nargs='+',help='list of file paths / text file containing paths to sharp images')
parser.add_argument('--synthetic_blur_paths', type=str, nargs='+',help='list of file paths/text file containing paths to blurred images')
parser.add_argument('--blur_type', type=str, default='blur',choices=['blur', 'gamma', 'synthetic'],help='Which blur folder to use')
parser.add_argument('--lambda_freq', type=float, default=1.0, help='weight for frequency-domain loss')
parser.add_argument('--num_feat', type=int, default=16, help='number of channels in attention branches')                   
parser.add_argument('--lambda_cycle', type=float, default=1.0, help='weight for cycle consistency loss')
parser.add_argument('--lambda_adv', type=float, default=0.2, help='weight for adversarial loss')
parser.add_argument('--lambda_adv_spatial', type=float, default=0.1, help='Weight for spatial adversarial loss')
parser.add_argument('--lambda_adv_freq', type=float, default=0.1, help='Weight for frequency adversarial loss')
parser.add_argument("--clip_grad", type=float, default=1.0, help="max norm for gradient clipping (applied to G and D)")
parser.add_argument('--freq_disc_depth', type=int, default=3,help='number of layers in frequency discriminator')
parser.add_argument('--freq_disc_kernel', type=int, default=3,help='kernel size for frequency discriminator convs')
parser.add_argument('--freq_disc_feats', type=int, default=64,help='base feature channels for frequency discriminator')
parser.add_argument('--use_nonlocal_attention', type=bool, default=False,help='use non-local spatial/channel attention refinement')
parser.add_argument('--milestones_G', nargs='+', type=int, default=[500,700,900])
parser.add_argument('--milestones_D', nargs='+', type=int, default=[500,700,900])
parser.add_argument('--gamma_G', type=float, default=0.5)
parser.add_argument('--gamma_D', type=float, default=0.5)
parser.add_argument('--lambda_lpips', type=float, default=0.5, help='Weight for LPIPS perceptual loss')
parser.add_argument('--lambda_grad', type=float, default=0.5, help='Weight for gradient sharpness loss')
parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='Weight for perceptual loss')
parser.add_argument('--lambda_neighbor', type=float, default=0.0, help='Weight for neighbor similarity loss')
parser.add_argument('--w_nonlocal', type=float, default=0.8,help='Weight for nonlocal attention branch')
parser.add_argument('--lambda_featmatch', type=float, default=0.2,help='Weight for feature matching loss')
parser.add_argument('--lambda_r1', type=float, default=10.0,help='Weight for R1 gradient penalty on real samples (discriminator regularization)')
parser.add_argument('--scheduler_D', type=str, default='warm_multistep',help='Learning rate scheduler type for discriminator (separate from generator)')
parser.add_argument('--log_adv_weights', action='store_true',help='Log adversarial component weights (patch, freq, featmatch) to TensorBoard each epoch')
parser.add_argument('--lambda_edge', type=float, default=0.1,help='Weight for edge loss term')
parser.add_argument("--reset_discriminator",action="store_true",help="Reset and reinitialize the discriminator at startup")


# Device specs
group_device = parser.add_argument_group('Device specs')
group_device.add_argument('--seed', type=int, default=-1, help='random seed')
group_device.add_argument('--num_workers', type=int, default=8, help='the number of dataloader workers')
group_device.add_argument('--device_type', type=str, choices=('cpu', 'cuda'), default='cuda', help='device to run models')
group_device.add_argument('--device_index', type=int, default=0, help='device id to run models')
group_device.add_argument('--n_GPUs', type=int, default=1, help='the number of GPUs for training')
group_device.add_argument('--distributed', type=str2bool, default=False, help='use DistributedDataParallel instead of DataParallel for better speed')
group_device.add_argument('--launched', type=str2bool, default=False, help='identify if main.py was executed from launch.py. Do not set this to be true using main.py.')

group_device.add_argument('--master_addr', type=str, default='127.0.0.1', help='master address for distributed')
group_device.add_argument('--master_port', type=int2str, default='8023', help='master port for distributed')
group_device.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
group_device.add_argument('--init_method', type=str, default='env://', help='distributed init method URL to discover peers')
group_device.add_argument('--rank', type=int, default=0, help='rank of the distributed process (gpu id). 0 is the master process.')
group_device.add_argument('--world_size', type=int, default=1, help='world_size for distributed training (number of GPUs)')

# Data specs
group_data = parser.add_argument_group('Data specs')
group_data.add_argument('--data_root', type=str, default=os.path.expanduser('~/GOPRO_LARGE/'), help='dataset root location')
group_data.add_argument('--dataset', type=str, default=None, help='training/validation/test dataset name, has priority if not None')
group_data.add_argument('--data_train', type=str, default='GOPRO_Large', help='training dataset name')
group_data.add_argument('--data_val', type=str, default=None, help='validation dataset name')
group_data.add_argument('--data_test', type=str, default='GOPRO_Large', help='test dataset name')
group_data.add_argument('--blur_key', type=str, default='blur_gamma', choices=('blur', 'blur_gamma'), help='blur type from camera response function for GOPRO_Large dataset')
group_data.add_argument('--rgb_range', type=int, default=255, help='RGB pixel value ranging from 0')

# Model specs
group_model = parser.add_argument_group('Model specs')
group_model.add_argument('--model', type=str, default='TraUNetGenerator', help='model architecture (TraUNetGenerator)')
group_model.add_argument('--pretrained', type=str, default='', help='pretrained model location')
group_model.add_argument('--gaussian_pyramid', type=str2bool, default=False, help='gaussian pyramid input/target')
group_model.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
group_model.add_argument('--kernel_size', type=int, default=5, help='size of conv kernel')
group_model.add_argument('--downsample', type=str, choices=('Gaussian', 'bicubic', 'stride'), default='Gaussian', help='input pyramid generation method')
group_model.add_argument('--disc_n_feats', type=int, default=64, help='base number of feature maps for discriminator')
group_model.add_argument('--use_spectral_norm', type=str2bool, default=True, help='Use spectral normalization in discriminator')
group_model.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test(single | half)')
group_model.add_argument('--use_sa', type=str2bool, default=True, help='enable all attention modules (EdgeAware, Spixel, DeblurUnSAM)')
group_model.add_argument('--use_edge_attention', type=str2bool, default=True, help='enable EdgeAwareAttention in generator')
group_model.add_argument('--use_spixel_attention', type=str2bool, default=True, help='enable EdgeSpixelAttention in generator')
group_model.add_argument('--use_unsam_attention', type=str2bool, default=True, help='enable DeblurUnSAM attention in generator')
group_model.add_argument('--out_activation', type=str, default='sigmoid', choices=('sigmoid', 'tanh'), help='output activation function for generator')
group_model.add_argument('--use_freq_disc', type=str2bool, default=True, help='enable frequency-domain discriminator')
group_model.add_argument('--disc_kernel_size', type=int, default=3, help='kernel size for discriminator convolutions')
group_model.add_argument('--n_scales', type=int, default=1, help='number of scales in the model')
group_model.add_argument('--w_edge', type=float, default=1.0, help='initial weight for Edge attention')
group_model.add_argument('--w_spixel', type=float, default=1.0, help='initial weight for Spixel attention')
group_model.add_argument('--w_unsam', type=float, default=1.0, help='initial weight for UnSAM attention')
group_model.add_argument('--disc_n_scales', type=int, default=3, help='number of scales in discriminator')
# AMP specs
group_amp = parser.add_argument_group('AMP specs')
group_amp.add_argument('--amp', type=str2bool, default=False, help='use automatic mixed precision training')
group_amp.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')

# Training specs
group_train = parser.add_argument_group('Training specs')
group_train.add_argument('--patch_size', type=int, default=0, help='training patch size')
group_train.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
group_train.add_argument('--split_batch', type=int, default=1, help='split a minibatch into smaller chunks')
group_train.add_argument('--augment', type=str2bool, default=True, help='train with data augmentation')

# Testing specs
group_test = parser.add_argument_group('Testing specs')
group_test.add_argument('--validate_every', type=int, default=10, help='do validation at every N epochs')
group_test.add_argument('--test_every', type=int, default=10, help='do test at every N epochs')
group_test.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
group_test.add_argument('--val_patch_size', type=int, default=0,help='crop size for validation images (0 = full resolution)')
group_test.add_argument('--metric',type=str,default='PSNR,SSIM',help='evaluation metrics to use, separated by commas (e.g. PSNR,SSIM,LPIPS,NIQE,BRISQUE)')

# Action specs
group_action = parser.add_argument_group('Source behavior')
group_action.add_argument('--do_train', type=str2bool, default=True, help='do train the model')
group_action.add_argument('--do_validate', type=str2bool, default=True, help='do validate the model')
group_action.add_argument('--do_test', type=str2bool, default=True, help='do test the model')
group_action.add_argument('--demo', type=str2bool, default=False, help='demo')
group_action.add_argument('--demo_input_dir', type=str, default='', help='demo input directory')
group_action.add_argument('--demo_output_dir', type=str, default='', help='demo output directory')

# Optimization specs
group_optim = parser.add_argument_group('Optimization specs')
group_optim.add_argument('--lr', type=float, default=1e-4, help='learning rate')
group_optim.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for discriminator')
group_optim.add_argument('--milestones', type=int, nargs='+', default=[500, 750, 900], help='learning rate decay per N epochs')
group_optim.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
group_optim.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSProp)')
group_optim.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
group_optim.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='ADAM betas')
group_optim.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon')
group_optim.add_argument('--weight_decay', type=float, default=0, help='weight decay')
group_optim.add_argument('--resume_optimizer', type=str2bool, default=True, help='Resume optimizer state from checkpoint')
group_optim.add_argument('--scheduler',type=str,default='step',choices=('step', 'cosine', 'warm_multistep', 'none'),help='learning rate scheduler to use (step | cosine | warm_multistep | none)')
group_optim.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs for warm_multistep scheduler')
group_optim.add_argument('--scale', type=float, default=10.0, help='warmup LR start factor: LR begins at base_lr/scale')

# Logging
group_log = parser.add_argument_group('Logging specs')
group_log.add_argument('--save_dir', type=str, default='', help='subdirectory to save experiment logs')
group_log.add_argument('--start_epoch', type=int, default=-1, help='(re)starting epoch number')
group_log.add_argument('--end_epoch', type=int, default=600, help='ending epoch number')
group_log.add_argument('--load_epoch', type=int, default=-1, help='epoch number to load model (start_epoch-1 for training, start_epoch for testing)')
group_log.add_argument('--save_every', type=int, default=10, help='save model/optimizer at every N epochs')
group_log.add_argument('--save_results', type=str, default='part', choices=('none', 'part', 'all'), help='save none/part/all of result images')
group_log.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
group_log.add_argument('--resume_dir', type=str, default='', help='Path to checkpoint directory')
group_log.add_argument('--manual_load_epoch', type=int, default=-1, help='Epoch to manually load model from')

# Debugging
group_debug = parser.add_argument_group('Debug specs')
group_debug.add_argument('--stay', type=str2bool, default=False, help='stay at interactive console after trainer initialization')

parser.add_argument('--template', type=str, default='', help='argument template option')

args, unknown = parser.parse_known_args()

if args.device_type == "cuda" and torch.cuda.is_available():
    args.device = f"cuda:{args.device_index}"
else:
    args.device = "cpu"

template.set_template(args)

args.data_root = os.path.expanduser(args.data_root) 
print("Resolved data_root:", args.data_root)
print("Exists:", os.path.exists(args.data_root))

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if not hasattr(args, 'save_dir') or args.save_dir == '':
    if args.resume and args.resume_dir != '':
        args.save_dir = os.path.basename(os.path.normpath(args.resume_dir))
    else:
        args.save_dir = now
    args.save_dir = os.path.join('../experiment', args.save_dir)

os.makedirs(args.save_dir, exist_ok=True)

# Resume logic
model_dir = os.path.join(args.resume_dir, 'models')
model_prefix = 'model-'

if args.resume:
    if os.path.exists(model_dir):
        def is_valid_checkpoint(path):
            try:
                torch.load(path, weights_only=True)
                return True
            except Exception:
                return False

        model_list = [name for name in os.listdir(model_dir) if name.startswith(model_prefix) and name.endswith('.pt')]
        model_list = [name for name in model_list if is_valid_checkpoint(os.path.join(model_dir, name))]

        if model_list:
            last_epoch = max([int(re.findall("\\d+", name)[0]) for name in model_list])
            args.manual_load_epoch = last_epoch
            args.start_epoch = last_epoch + 1
            print(f"[INFO] Auto-resuming from epoch {last_epoch}")
        else:
            print(f"[WARNING] No valid checkpoints found in '{model_dir}'. Starting from scratch.")
            args.start_epoch = 1
    else:
        print(f"[WARNING] Resume directory '{model_dir}' not found. Starting from scratch.")
        args.start_epoch = 1
elif args.start_epoch < 0:
    args.start_epoch = 1
elif args.start_epoch == 0:
    if args.rank == 0:
        shutil.rmtree(args.save_dir, ignore_errors=True)
    os.makedirs(args.save_dir, exist_ok=True)
    args.start_epoch = 1

if args.load_epoch < 0:
    args.load_epoch = args.start_epoch - 1

if args.pretrained:
    if args.start_epoch <= 1:
        args.pretrained = os.path.join('../experiment', args.pretrained)
    else:
        print(f"Starting from epoch {args.start_epoch}! Ignoring pretrained model path...")
        args.pretrained = ''

argname = os.path.join(args.save_dir, 'args.pt')
argname_txt = os.path.join(args.save_dir, 'args.txt')
if args.start_epoch > 1 and os.path.exists(argname):
    args_old = torch.load(argname)
    load_list = ['patch_size', 'batch_size', 'rgb_range', 'blur_key', 'n_scales', 'n_resblocks', 'n_feats']
    for arg_part in load_list:
        if arg_part in vars(args_old):
            vars(args)[arg_part] = vars(args_old)[arg_part]

if args.dataset is not None:
    args.data_train = args.dataset
    args.data_val = args.dataset
    args.data_test = args.dataset

if args.data_val is None:
    args.do_validate = False

if args.demo_input_dir:
    args.demo = True

if args.demo:
    assert os.path.basename(args.save_dir) != now, 'You should specify pretrained directory by setting --save_dir SAVE_DIR'
    args.data_train = ''
    args.data_val = ''
    args.data_test = ''
    args.do_train = False
    args.do_validate = True
    args.validate_every = 10
    args.do_test = False
    assert len(args.demo_input_dir) > 0, 'Please specify demo_input_dir!'
    args.demo_input_dir = os.path.expanduser(args.demo_input_dir)
    if args.demo_output_dir:
        args.demo_output_dir = os.path.expanduser(args.demo_output_dir)
    args.save_results = 'all'

if args.amp:
    args.precision = 'single'

if args.seed < 0:
    args.seed = int(time.time())

# save arguments
if args.rank == 0:
    torch.save(args, argname)
    with open(argname_txt, 'a') as file:
        file.write(f'execution at {now}\n')

        for key in args.__dict__:
            file.write(key + ': ' + str(args.__dict__[key]) + '\n')
        file.write('\n')

if args.device_type == 'cuda' and not torch.cuda.is_available():
    raise Exception("GPU not available!")

if not args.distributed:
    args.rank = 0

def setup(args):
    cudnn.benchmark = True
    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        args.device_index = args.rank
        args.world_size = args.n_GPUs
        dist.init_process_group(args.dist_backend, init_method=args.init_method, rank=args.rank, world_size=args.world_size)
    args.device = torch.device(args.device_type, args.device_index) if args.device_type == 'cuda' else torch.device(args.device_type)
    args.dtype = torch.float32
    args.dtype_eval = torch.float32 if args.precision == 'single' else torch.float16
    torch.manual_seed(args.seed)
    if args.device_type == 'cuda':
        torch.cuda.set_device(args.device)
        if args.rank == 0:
            torch.cuda.manual_seed_all(args.seed)
    return args

def cleanup(args):
    if args.distributed:
        dist.destroy_process_group()


