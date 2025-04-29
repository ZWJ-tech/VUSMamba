import os
import h5py
import numpy as np
import torch
# from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage as ndi
# from scipy import stats

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    for zz in range(image.shape[0]):
        image[zz,:,:] = np.rot90(image[zz,:,:], k)
        label[zz,:,:] = np.rot90(label[zz,:,:], k)
        axis = np.random.randint(0, 2)
        image[zz,:,:] = np.flip(image[zz,:,:], axis=axis).copy()
        label[zz,:,:] = np.flip(label[zz,:,:], axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    for zz in range(image.shape[0]):
        image[zz,:,:] = ndimage.rotate(image[zz,:,:], angle, order=0, reshape=False)#旋转
        label[zz,:,:] = ndimage.rotate(label[zz,:,:], angle, order=0, reshape=False)
    return image, label

def scale_intensity_ranged(data, a_min=112, a_max=1000, b_min=0.0, b_max=1.0, clip=True):
    scaled_data = (data - a_min) / (a_max - a_min)  # 缩放到[0, 1]范围内

    if clip:
        scaled_data = np.clip(scaled_data, 0, 1)  # 裁剪到[0, 1]范围内

    scaled_data = scaled_data * (b_max - b_min) + b_min  # 缩放到目标范围

    return scaled_data

def norm_data(data):
    data1 = data - np.min(data) 
    data = data1 * 1.0 / (np.max(data) - np.min(data) )
    return data

# def simple_norm(img, a, b, m_high=-1, m_low=-1):
#     idx = np.ones(img.shape, dtype=bool)
#     if m_high>0:
#         idx = np.logical_and(idx, img<m_high)
#     if m_low>0:
#         idx = np.logical_and(idx, img>m_low)
#     img_valid = img[idx]
#     m,s = stats.norm.fit(img_valid.flat)
#     strech_min = max(m - a*s, img.min())
#     strech_max = min(m + b*s, img.max())
#     img[img>strech_max]=strech_max
#     img[img<strech_min]=strech_min
#     img = (img- strech_min)/(strech_max - strech_min)
#     return img

def background_sub(img, r):
    struct_img_smooth = ndi.gaussian_filter(img, sigma=r, mode='nearest', truncate=3.0)
    struct_img_smooth_sub = img - struct_img_smooth
    struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min())/(struct_img_smooth_sub.max()-struct_img_smooth_sub.min())
    return struct_img

def gamma(image, c, v):
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if 300 <= image[i,j] <= 680:
                value = image[i,j] / 255
                new_image[i,j] = c * np.power(value, v) * 255
            else:
                new_image[i,j] = c * image[i,j]
    output_image = np.uint16(new_image + 0.5)

    return output_image

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_c1, image_c2 = sample['image_c1'], sample['image_c2']
        # image[image<=112] = 0
        # image = background_sub(image, 2)
        image_c1 = scale_intensity_ranged(image_c1) 
        image_c2 = scale_intensity_ranged(image_c2) 
        # contrast = random.random()
        # new_image = np.zeros_like(image)
        # if contrast < 0.5:
        #     for zz in range(image.shape[0]):
        #         new_image[zz,:,:] = gamma(image[zz,:,:], 1, 2) 
        image_c1 = np.expand_dims(image_c1, axis=0)
        image_c2 = np.expand_dims(image_c2, axis=0)
        image_c1 = torch.from_numpy(image_c1.astype(np.float32))
        image_c2 = torch.from_numpy(image_c2.astype(np.float32))
        sample = {'image_c1': image_c1, 'image_c2': image_c2}

        return sample

class VISoR_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image_c1, image_c2 = data['image_c1'], data['image_c2']
            # label[label==255] = 1
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            label[label==255] = 1
            # new_image = np.zeros_like(image)
            # for zz in range(image.shape[0]):
            #     new_image[zz,:,:] = gamma(image[zz,:,:], 1, 2) 
            # image[image<=112] = 0
            # image = background_sub(image, 2)
            # image = norm_data(image) 
        sample = {'image_c1': image_c1, 'image_c2': image_c2}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def standard_data(input):
     
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    standard = (input - mean) / std
    standard_input = torch.from_numpy(standard)
    return standard_input

class Points_dataset(Dataset):
    def __init__(self, src_path, ref_path, src_list_path, ref_list_path, src_split, ref_split):
        super().__init__()
        self.src_path = src_path
        self.ref_path = ref_path
        self.src_split = src_split
        self.ref_split = ref_split
        self.src_sample_list = open(os.path.join(src_list_path, self.src_split+'.txt')).readlines()
        self.ref_sample_list = open(os.path.join(ref_list_path, self.ref_split+'.txt')).readlines()

    def __getitem__(self, idx):
        src_slice_name = self.src_sample_list[idx].strip('\n')
        coordinates = []
        with open(os.path.join(self.src_path, src_slice_name+'.txt'), 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) == 3:
                    x, y, z = parts
                    coordinate = (x, y, z)
                    coordinates.append(coordinate)
        
        coordinates_array = np.array(coordinates).astype(np.float32)
        src = standard_data(coordinates_array)

        ref_slice_name = self.ref_sample_list[idx].strip('\n')
        ref_coordinates = []
        with open(os.path.join(self.ref_path, ref_slice_name+'.txt'), 'r') as ref_file:
            for ref_line in ref_file:
                ref_parts = ref_line.strip().split(' ')
                if len(ref_parts) == 3:
                    ref_x, ref_y, ref_z = ref_parts
                    ref_coordinate = (ref_x, ref_y, ref_z)
                    ref_coordinates.append(ref_coordinate)
        
        ref_coordinates_array = np.array(ref_coordinates).astype(np.float32)
        ref = standard_data(ref_coordinates_array)

        sample = {'src': src, 'ref': ref}

        return sample

    def __len__(self):
        return len(self.src_sample_list)