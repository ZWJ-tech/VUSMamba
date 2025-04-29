import numpy as np
import torch
from numpy.random import randint

def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, z, h, w = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x

def patch_exchange(x, patch_shape=(30, 30, 30), patch_random_num=8, max_mov_dist=(6, 60, 60)):
    c, z, h, w = x.size()
    rand_nums = randint(0, patch_random_num)
    for i in range(rand_nums):
        start_x = randint(0, h - patch_shape[2] + 1)
        start_y = randint(0, w - patch_shape[1] + 1)
        start_z = randint(0, z - patch_shape[0] + 1)

        max_end_x = min(h - patch_shape[2], start_x + max_mov_dist[2])
        max_end_y = min(w - patch_shape[1], start_y + max_mov_dist[1])
        max_end_z = min(z - patch_shape[0], start_z + max_mov_dist[0])

        end_x = randint(abs(start_x - max_mov_dist[2]), max_end_x + 1)
        end_y = randint(abs(start_y - max_mov_dist[1]), max_end_y + 1)
        end_z = randint(abs(start_z - max_mov_dist[0]), max_end_z + 1)

        original_value = x[:, start_z:start_z + patch_shape[0], start_x:start_x + patch_shape[0], start_y:start_y + patch_shape[0]]
        target_value = x[:, end_z:end_z + patch_shape[0], end_x:end_x + patch_shape[2], end_y:end_y + patch_shape[1]]

        x[:, start_z:start_z + patch_shape[0], start_x:start_x + patch_shape[0], start_y:start_y + patch_shape[0]] = target_value
        x[:, end_z:end_z + patch_shape[0], end_x:end_x + patch_shape[2], end_y:end_y + patch_shape[1]] = original_value

    return x

def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().to(args.device)
    x_rot = torch.zeros(img_n).long().to(args.device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation

    return x_aug, x_rot

def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        # x_aug[i] = patch_rand_drop(args, x_aug[i])
        x_aug[i] = patch_exchange(x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            # x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
            x_aug[i] = patch_exchange(x_aug[i])

    return x_aug

if __name__ == "__main__":

    data = np.random.randint(0,256, (2, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    # x = rot_rand(data)
    x1_augment = aug_rand(data)