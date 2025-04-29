import argparse
import os
import sys
import logging
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from brainseg_load import VISoR_dataset, RandomGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
from model.swin_transform import swin_tiny_patch4_window8_256
from model.unet3d import UNet3D
from model.swinumamba import get_swin_umamba
from model.swinunetr import SwinUNETR
from model.cpnet import CPnet3D
from loss import DiceLoss, MemoryEfficientSoftDiceLoss
from torch.autograd import Variable
from utils import ConfusionMatrix, weights_init
from torch import distributed as dist

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--data_path", default="/home/weijie/H3C/model_train_test/train_seg_data", type=str, help="data path")
parser.add_argument("--list_dir", default="/home/weijie/MaskFormer3d/seg_lists", type=str, help="list path")
parser.add_argument("--img_size", default=256, type=int, help="size of input image")
parser.add_argument("--model_in_chans", default=1, type=int, help="Input channel to the model")
parser.add_argument("--patch_size", default=4, type=int, help="the patch size in 3D-HSFormer")
parser.add_argument("--window_size", default=8, type=int, help="the window size in 3D-HSFormer")
parser.add_argument("--embed_dim", default=96, type=int, help="the embed dim in 3D-HSFormer")
parser.add_argument(
    "--pretrained_dir", default="/home/weijie/MaskFormer3d/celltype_checkpoint", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name",
    default="pretrain_epoch_99.pth",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--logdir", default="train_seg", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--max_epochs", default=30, type=int, help="number of training epochs")
parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--feature_size", default=96, type=int, help="embedding size")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--lr", default=1e-03, type=float, help="learning rate")
parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
parser.add_argument("--loss_type", default="SSL", type=str)
parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
parser.add_argument("--resume_ckpt", default=False, help="resume training from pretrained checkpoint")
parser.add_argument("--use_ssl_pretrained", default=False, help="use self-supervised pretrained weights")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
parser.add_argument("--noamp", default=True, help="do NOT use amp for training")
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
parser.add_argument("--save_checkpoint", default="/data/weijie/Celltype_checkpoint/swinunetr_vglut_vgat", help="url used to set up distributed training")
parser.add_argument("--best_iou", default=90, help="best IOU")

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    best_iou = args.best_iou
    
    logging.basicConfig(filename=args.logdir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    train_data = VISoR_dataset(base_dir=args.data_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_data = VISoR_dataset(base_dir=args.data_path, list_dir=args.list_dir, split="valid",
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

    pretrained_dir = args.pretrained_dir
    # model = swin_tiny_patch4_window8_256(num_classes=args.out_channels)
    # model = get_swin_umamba(num_classes=args.out_channels)
    # model = UNet3D(in_channel=1, n_classes=2)
    # model = CPnet3D(nbase=[1, 32, 64, 128, 256], out_classes=2, sz=3, residual_on=False, style_on=False, concatenation=True)
    model = SwinUNETR(args)
    model.apply(weights_init)
    
    model.to(device)
    model.train()

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("/data/weijie/Celltype_checkpoint/swinunetr_log/model_final_epoch.pt")
            state_dict = model_dict["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.out_channels)
    # dice_loss = MemoryEfficientSoftDiceLoss(batch_dice=True, smooth=1e-5, ddp=dist.is_available() and dist.is_initialized())
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.05) 
    iter_num = 0
    validate_every_n_epoch = 2
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)   
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            image_batch, label_batch = Variable(image_batch), Variable(label_batch)
            pred = model(image_batch)
            # pred = pred['out']
            loss_ce = criterion(pred, label_batch[:].long())
            # loss = criterion(pred, label_batch)
            loss_dice = dice_loss(pred, label_batch, softmax=True)
            # loss_dice = dice_loss(pred, label_batch)
            # loss_cldice = cldice_loss(label_batch, pred)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            lr_ = optimizer.param_groups[0]["lr"]
            lr_ = args.lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : loss : %f, loss_ce : %f' % (iter_num, loss.item(), loss_ce.item()))

        if epoch_num % validate_every_n_epoch ==0:
            confmat = ConfusionMatrix(args.out_channels)
            val_loader = tqdm(valid_loader, desc="Validate")
            val_iter_num = 0
            model.eval()
            for i, val_data in enumerate(val_loader):
                val_image, val_label = val_data['image'], val_data['label']
                val_image, val_label = val_image.to(device), val_label.to(device)
                with torch.no_grad():
                    val_out = model(val_image)
                    # val_out = val_out['out']
                    val_out = torch.softmax(val_out, dim=1)
                    confmat.update(val_label.squeeze().flatten(), torch.argmax(val_out, dim=1).flatten())
                    # val_loss_ce = criterion(val_out, val_label[:].long())
                    # val_loss_dice = dice_loss(val_out, val_label, softmax=False)
                    # val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice
                val_iter_num = val_iter_num + 1
                logging.info(confmat)
            _, _, iu = confmat.compute()
            mean_iu = iu.mean().item() * 100
            if mean_iu > best_iou:
                best_iou += 1
                os.makedirs(args.save_checkpoint, exist_ok=True)
                save_mode_path = os.path.join(args.save_checkpoint, 'epoch_best_{}.pth'.format(epoch_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if epoch_num > 1000 and mean_iu < 51:
                break

        save_interval = 5  # int(max_epoch/6)

        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(args.save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            os.makedirs(args.save_checkpoint, exist_ok=True)
            save_mode_path = os.path.join(args.save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"

if __name__ == "__main__":
    main()