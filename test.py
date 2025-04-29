import argparse
import logging
import os
import sys
import numpy as np
import math
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from brainseg_load import VISoR_dataset
from utils import test_single_volume
from model.swin_transform import swin_tiny_patch4_window8_256, swin_base_patch4_window8_256
from model.swinumamba import get_swin_umamba
from model.unet3d import UNet3D
from model.cpnet import CPnet3D
from model.swinunetr import SwinUNETR

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/data/weijie/SST_test_30', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='VISoR', help='experiment_name')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--model_name', type=str,
                    default='unet', help='model_name')

parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')

args = parser.parse_args()

def inference(args, model, device, test_save_path=None):
    test_data = args.Dataset(base_dir=args.volume_path, split="test_dapi", list_dir=args.list_dir)

    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = []
    invalid_num = 0
    patch_csv_name = args.model_name + '_patch_results.csv'
    patch_csv_root = '/home/weijie/H3C/model_evl/swinunter/dapi/patch_metric_csv'
    os.makedirs(patch_csv_root, exist_ok=True)
    patch_csv_save_path = os.path.join(patch_csv_root, patch_csv_name)
    patch_list_name = ['Dice', 'Hd95', 'Jc', 'Sst']
    with open(patch_csv_save_path, 'w+', newline='')as f:
        patch_csv_write = csv.writer(f, dialect='excel')
        patch_csv_write.writerow(patch_list_name)
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            # h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            image, label = image.to(device), label.to(device)
            metric_i, confmat = test_single_volume(image, label, model, device, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                        test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            # metric_list += np.array(metric_i)
            patch_writelines = [metric_i[0][0], metric_i[0][1], metric_i[0][2], metric_i[0][3]]
            patch_csv_write.writerow(patch_writelines)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_jc %f mean_sst %f' % (i_batch, case_name, 
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
                        np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
            if np.mean(metric_i, axis=0)[0] == 0:
                pass
            else:
                metric_list.append(metric_i)
    confmat.reduce_from_all_processes()
    logging.info(confmat)
    # metric_mean_list = metric_list / (len(testloader) - invalid_num)
    metric_mean_list = np.mean(metric_list, axis=0)
    metric_std_list = np.std(metric_list, axis=0)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_jc %f mean_sst %f' % (i, metric_mean_list[i-1][0], metric_mean_list[i-1][1], 
                                                                                            metric_mean_list[i-1][2], metric_mean_list[i-1][3]))
                                                                                      
    performance = np.mean(metric_mean_list, axis=0)[0]
    mean_hd95 = np.mean(metric_mean_list, axis=0)[1]
    mean_jc = np.mean(metric_mean_list, axis=0)[2]
    mean_sst = np.mean(metric_mean_list, axis=0)[3]
    Mean_list = [performance, mean_hd95, mean_jc, mean_sst]

    std_dice = metric_std_list[0][0]
    std_hd95 = metric_std_list[0][1]
    std_jc = metric_std_list[0][2]
    std_sst = metric_std_list[0][3]
    Std_list = [std_dice, std_hd95, std_jc, std_sst]

    ste_dice = std_dice / math.sqrt((len(testloader) - invalid_num))
    ste_hd95 = std_hd95 / math.sqrt((len(testloader) - invalid_num))
    ste_jc = std_jc / math.sqrt((len(testloader) - invalid_num))
    ste_sst = std_sst / math.sqrt((len(testloader) - invalid_num))
    Ste_list = [ste_dice, ste_hd95, ste_jc, ste_sst]

    csv_name = args.model_name + '_results.csv'
    csv_root = '/home/weijie/H3C/model_evl/swinunter/dapi/metric_csv'
    os.makedirs(csv_root, exist_ok=True)
    csv_save_path = os.path.join(csv_root, csv_name)
    list_name = ['Metric_name','Mean', 'Std', 'Ste']
    metric_name = ['Dice', 'Hd95', 'Jc', 'Sst']
    with open(csv_save_path, 'w+', newline='')as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(list_name)
        for num in range(len(metric_name)):
            writelines = [metric_name[num], Mean_list[num], Std_list[num], Ste_list[num]]
            csv_write.writerow(writelines)

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_jc: %f mean_sst: %f' % (performance, mean_hd95, mean_jc, mean_sst))
    logging.info('Testing performance in best val model: std_dice : %f std_hd95 : %f std_jc: %f std_sst: %f' % (std_dice, std_hd95, std_jc, std_sst))
    logging.info('Testing performance in best val model: ste_dice : %f ste_hd95 : %f ste_jc: %f ste_sst: %f' % (ste_dice,ste_hd95, ste_jc, ste_sst))

    return "Testing Finished!"

if __name__ == "__main__":

    dataset_config = {
        'VISoR': {
            'Dataset': VISoR_dataset,
            'volume_path': '/home/weijie/H3C/Cell_type_test_data/20240117_MHW_JQ_WT1/2.0/Annotation_data/test_data/test_dapi_62',
            'list_dir': '/home/weijie/MaskFormer3d/seg_lists',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.exp = dataset_name + str(args.img_size)
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    args.is_savenii = False

    # name the same snapshot defined in train script!
    snapshot_path = "/data/weijie/Celltype_checkpoint/ThreeD-HSFormer_v4"
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    net = swin_tiny_patch4_window8_256(num_classes=2).to(device)
    # net = get_swin_umamba(num_classes=2).to(device)
    # net = UNet3D(in_channel=1, n_classes=2).to(device)
    # net = CPnet3D(nbase=[1, 32, 64, 128, 256], out_classes=2, sz=3, residual_on=False, style_on=False, concatenation=True).to(device)
    # net = SwinUNETR(args).to(device)

    # model_dict = torch.load("/home/weijie/MaskFormer3d/log/model_final_epoch.pt")
    # state_dict = model_dict["state_dict"]
    snapshot = os.path.join(snapshot_path, 'epoch_best_4.pth')
    # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    # net.load_state_dict(state_dict, strict=False)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '/home/weijie/H3C/model_evl/Ours'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, device, test_save_path)