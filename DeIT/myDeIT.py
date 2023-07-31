import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from torchvision import datasets
from torchvision import transforms # for simplifying the transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import torch.backends.cudnn as cudnn
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss
#for models
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.optim import create_optimizer
from timm.data import Mixup


from DeIT_Helper import DistillationLoss, train_one_epoch, evaluate
from datasets import build_dataset

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import utils
import models
import models_v2

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    #important args
    parser.add_argument('--data-path', default="/home/vinso/vscode-projects/repos/ImageClassificationData/food101", type=str, # will download in path if not found
                        help='dataset path')
    parser.add_argument('--data-set', default='FOOD', choices=['FOOD', 'CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset')
    parser.add_argument('--resume', default='/home/vinso/vscode-projects/myViT/TorchClassifier/DeIT/outputs/best_checkpoint_deit_base_patch16_224.pth', help='path to checkpoint if available')

    #hyper params
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=5, type=int)

    #models used
    # deit_base_patch16_224
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    # https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
    parser.add_argument('--finetune', default="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth", help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='/home/vinso/vscode-projects/myViT/TorchClassifier/DeIT/outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)




    #hyperparameters
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation Params
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')

    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--attn-only', action='store_true') 

    # Dataset parameters
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)
    return parser


def main(args):
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = 1

    cudnn.benchmark = True


    #load datasets and create data_loaders
    train_data, args.nb_classes = build_dataset(is_train=True, args=args)
    val_data, _ = build_dataset(is_train=False, args=args)

    #get samplers
    sampler_train = torch.utils.data.RandomSampler(train_data)
    sampler_val = torch.utils.data.SequentialSampler(val_data)

    # get data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    #making model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes, # 101
        drop_rate=args.drop, # 0.0
        drop_path_rate=args.drop_path, # 0.1
        drop_block_rate=None,
        img_size=args.input_size # 224
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            # load checkpoint from web
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            # load checkpoint from file
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            #check if key in checkpoint model and check if model shape differs from key in state_dict
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                # if shape is different, remove checkpoint
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # retrieves tensor from checkpoint model
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        # Determine dimensions and number of patches 
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        # reshape and and concatenate tokens 
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # update pos_embed with interpolated positions
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
            
    if args.attn_only: 
        # loops through all parameters 
        for name_p,p in model.named_parameters():
            #check if parameter in model contains '.attn.' substring
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                #freezes param -> no training on this param
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    # Smoothes out weight updating
    # uses set of averaged model parameters that update over time
    # assigns higher weights to more recent parameters
    # gradually diminishes the influence of older parameters
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, 
        # and AMP but before SyncBN and DDP wrapper
        # use timm function
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    # get sum of elements in model params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # if lr is unscaled -> false
    if not args.unscale_lr:
        # calculate linear scaled learing rate
        linear_scaled_lr = args.lr * args.batch_size * world_size / 512.0
        # set args lr to linear scaled lr
        args.lr = linear_scaled_lr

    #create optimizer, lr scheduler and loss function
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = LabelSmoothingCrossEntropy()

    # for using distillation loss
    teacher_model = None

    
    # if we pass in distillation type
    ''' not used for now
    '''
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    # # wrap the criterion in our custom DistillationLoss, which
    # # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )

    # specify output dir
    output_dir = Path(args.output_dir)
    if args.resume: #resume with previous checkpoint
        if args.resume.startswith('https'):
            #load checkpoint from web
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else: 
            #load checkpoint from file
            checkpoint = torch.load(args.resume, map_location='cpu')
        #model = load checkpoint
        model_without_ddp.load_state_dict(checkpoint['model'])
        #if no args.eval
        #check if optimizer, lr_scheduler, and epoch are terms in checkpoint 
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            #set optimizer optimizer, lr_scheduler, and epoch
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema: # use ema
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint: # use custom loss scaler
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch) #step lr
    if args.eval: # if eval_only is true
        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
        return

    #training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    # loop through epochs
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn=None,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )
        
        #step lr_scheduler per epoch
        lr_scheduler.step(epoch)
        if args.output_dir:
            #if output dir is specified
            checkpoint_paths = [output_dir / 'checkpoint_' + args.model + '.pth']
            for checkpoint_path in checkpoint_paths:
                #save model and args
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             
        # get accuracy 
        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            #save best acc
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                #create checkpoint
                checkpoint_paths = [output_dir / 'best_checkpoint_' + args.model + '.pth']
                for checkpoint_path in checkpoint_paths:
                    # if best acc, save model
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)