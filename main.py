import argparse
import logging
import os
import sys
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_load import RandomGenerator, VISoR_dataset
from loss import Loss
# from model.swinunetr_ssl_head import SSLHead
from lr_scheduler import WarmupCosineSchedule
from model.ssl_head import SSLHead
from ops import aug_rand, rot_rand


def train(args, global_step, val_best, scaler):
    model.train()
    logging.basicConfig(filename=args.logdir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    loss_train = []
    loss_train_recon = []

    train_data = VISoR_dataset(base_dir=args.data_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_data = VISoR_dataset(base_dir=args.data_path, list_dir=args.list_dir, split="valid",
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=1)


    for step, batch in enumerate(train_loader):
        t1 = time()
        x_c1 = batch['image_c1'].to(args.device)
        x_c2 = batch['image_c2'].to(args.device)
        x1, rot1 = rot_rand(args, x_c1)
        x2, rot2 = rot_rand(args, x_c2)
        x1_augment = aug_rand(args, x1)
        x2_augment = aug_rand(args, x2)
        # x1_augment = x1_augment
        # x2_augment = x2_augment
        with autocast(enabled=args.amp):
            rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
            rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
            rot_p = torch.cat([rot1_p, rot2_p], dim=0)
            rots = torch.cat([rot1, rot2], dim=0)
            imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
            imgs = torch.cat([x1, x2], dim=0)
            loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
        loss_train.append(loss.item())
        loss_train_recon.append(losses_tasks[2].item())
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.lrdecay:
            scheduler.step()
        optimizer.zero_grad()
        if args.distributed:
            if dist.get_rank() == 0:
                    logging.info("Step:{}/{}, Loss:{:.6f}, Time:{:.6f}".format(global_step, args.num_steps, loss, time() - t1))
        else:
            logging.info("Step:{}/{}, Rot Loss:{:.6f}, Contrastive Loss:{:.6f}, Reconstruction Loss:{:.6f}, Time:{:.4f}".format(global_step, args.num_steps, 
                            losses_tasks[0], losses_tasks[1], losses_tasks[2], time() - t1))
        global_step += 1
        if args.distributed:
            val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
        else:
            val_cond = global_step % args.eval_num == 0
        if val_cond:
            val_loss, val_loss_recon, img_list = validation(args, test_loader=valid_loader)
            logging.info("Validation/loss_recon {} step/ {}".format(val_loss_recon, global_step))
            logging.info("train/loss_total {} step/ {}".format(np.mean(loss_train), global_step))
            logging.info("train/loss_recon {} step/ {}".format(np.mean(loss_train_recon), global_step))

            if val_loss_recon < val_best:
                val_best = val_loss_recon
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                logging.info(
                    "Model was saved ! Best Recon. Val Loss {:.6f}, Recon. Val Loss {:.6f}".format(
                        val_best, val_loss_recon
                    )
                )
            else:
                print(
                    "Model was not saved ! Best Recon. Val Loss: {:.6f} Recon. Val Loss: {:.6f}".format(
                        val_best, val_loss_recon
                    )
                )
    return global_step, loss, val_best

def validation(args, test_loader):
    model.eval()
    loss_val = []
    loss_val_recon = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            val_inputs_c1 = batch['image_c1'].to(args.device)
            val_inputs_c2 = batch['image_c2'].to(args.device)
            x1, rot1 = rot_rand(args, val_inputs_c1)
            x2, rot2 = rot_rand(args, val_inputs_c2)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            with autocast(enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
            loss_recon = losses_tasks[2]
            loss_val.append(loss.item())
            loss_val_recon.append(loss_recon.item())
            x_gt = x1.detach().cpu().numpy()
            x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
            xgt = x_gt[0][0][:, :, 48] * 255.0
            xgt = xgt.astype(np.uint8)
            x1_augment = x1_augment.detach().cpu().numpy()
            x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
            x_aug = x1_augment[0][0][:, :, 48] * 255.0
            x_aug = x_aug.astype(np.uint8)
            rec_x1 = rec_x1.detach().cpu().numpy()
            rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
            recon = rec_x1[0][0][:, :, 48] * 255.0
            recon = recon.astype(np.uint8)
            img_list = [xgt, x_aug, recon]
            print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_recon))

    return np.mean(loss_val), np.mean(loss_val_recon), img_list


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--data_path", default="/data/weijie/celltype_train_data", type=str, help="data path")
    parser.add_argument("--list_dir", default="/home/weijie/MaskFormer3d/lists", type=str, help="list path")
    parser.add_argument("--img_size", default=256, type=int, help="size of input image")
    parser.add_argument("--model_in_chans", default=1, type=int, help="Input channel to the model")
    parser.add_argument("--patch_size", default=4, type=int, help="the patch size in 3D-HSFormer")
    parser.add_argument("--window_size", default=8, type=int, help="the window size in 3D-HSFormer")
    parser.add_argument("--embed_dim", default=96, type=int, help="the embed dim in 3D-HSFormer")

    parser.add_argument("--logdir", default="/data/weijie/Celltype_checkpoint/cpnet_log", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=5000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-03, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", default=True, help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    args = parser.parse_args()
    logdir = args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False

    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:1"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    else:
        writer = None

    model = SSLHead(args)
    model.to(args.device)

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]

    loss_function = Loss(args.batch_size * args.sw_batch_size, args)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, best_val, scaler)

    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "/final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "/final_model.pth")

    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")
