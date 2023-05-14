
import os

from numpy import pi, unicode_
from torch._C import set_flush_denormal
import torch
from torchvision.transforms.transforms import Resize
from base import BaseDataLoader

import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf
from scipy import interpolate

from PIL import Image

from typing import Union, Tuple, List

from utils import get_downscaled_size
import torchvision

from glob import glob
from os.path import join as pjoin

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import numpy as np

def load_exr(path, channel = 3):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[..., :channel]
    # if use_opencv or channel == 1:
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., :3]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[..., :channel]
    # else:
    #     img = imageio.imread(path, "exr")[..., :channel]

    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 1000

    return img

def upsampling_mv(motion_vector):
    ratio_x = 2
    ratio_y = 2

    motion_vector[..., 1] = motion_vector[..., 1] * ratio_y
    motion_vector[..., 0] = motion_vector[..., 0] * ratio_x

    upsampled = cv2.resize(motion_vector, [motion_vector.shape[1]*ratio_y, motion_vector.shape[0]*ratio_x], interpolation=cv2.INTER_LINEAR)

    return upsampled

def MyCollate_fn(batch):

    view_list = batch[0][0]
    depth_list = batch[0][1]
    flow_list = batch[0][2]
    truth = batch[0][3]

    for i in range(1, len(batch)):

        for j in range(len(view_list)):
            view_list[j] = torch.cat([view_list[j], batch[i][0][j]], dim=0)
        
        for j in range(len(depth_list)):
            depth_list[j] = torch.cat([depth_list[j], batch[i][1][j]], dim=0)

        for j in range(len(flow_list)):
            flow_list[j] = torch.cat([flow_list[j], batch[i][2][j]], dim=0)
        
        truth = torch.cat([truth, batch[i][3]], dim=0)

    return view_list, depth_list, flow_list, truth


class NSRRDataLoader(BaseDataLoader):
    """
    Generate batch of data
    `for x_batch in data_loader:`
    `x_batch` is a list of 4 tensors, meaning `view, depth, flow, view_truth`
    each size is (batch x channel x height x width)
    """
    def __init__(self,
                 data_dir_list: list,
                 batch_size: int,
                 cropped_size: Union[Tuple[int, int], List[int], int] = None,
                 cropped_num: int = 1,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 ):
        dataset = NSRRDataset(data_dir_list,
                              cropped_size = cropped_size,
                              cropped_num = cropped_num
                              )
        super(NSRRDataLoader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             validation_split=validation_split,
                                             num_workers=num_workers,
                                             collate_fn=MyCollate_fn
                                             )


class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dir_list: list,
                 cropped_size: Union[Tuple[int, int], List[int], int] = (256, 256),
                 cropped_num: int = 1,
                 transform: nn.Module = None,
                 ):
        super(NSRRDataset, self).__init__()

        self.cropped_size = cropped_size
        self.cropped_num = cropped_num

        if transform is None:
            self.transform = tf.ToTensor()

        self.img_list = []
        for data_dir in data_dir_list:
            img_paths = glob(pjoin(data_dir, "PreTonemapHDRColor.*.exr"))
            tmp_list = []
            for path in img_paths:
                idx = int(os.path.basename(path).split('.')[1])
                tmp_list.append((data_dir, idx))
            tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=False)

            self.img_list += tmp_list
        
        self.data_list = []
        
        for i in range(len(self.img_list)):

            if(i==len(self.img_list)-4): # if current frame is the last img
                break   

            current_frame = self.img_list[i+4]
            pre_1, pre_2, pre_3, pre_4 = self.img_list[i+3], self.img_list[i+2], self.img_list[i+1], self.img_list[i]

            if current_frame[0]!=pre_1[0] or current_frame[0]!=pre_2[0] or current_frame[0]!=pre_3[0] or current_frame[0]!=pre_4[0]:
                continue

            self.data_list.append([current_frame, pre_1, pre_2, pre_3, pre_4])
                
    def __getitem__(self, index):
        # view
        # image_name = self.view_listdir[index]
        data = self.data_list[index]

        crop_l, crop_r, crop_t, crop_b = None, None, None, None


        view_list, depth_list, flow_list, truth_list = [], [], [], []
        # elements in the lists following the order: current frame i, pre i-1, pre i-2, pre i-3, pre i-4
        for frame in data:

            data_dir = frame[0]
            idx = frame[1]

            low_img_path = pjoin(data_dir, "PreTonemapHDRColor.{:04d}.exr".format(idx))
            high_img_path = pjoin(data_dir, "High+SSAA", "PreTonemapHDRColor64SPP.{:04d}.exr".format(idx))
            depth_img_path = pjoin(data_dir, "SceneDepth.{:04d}.exr".format(idx))
            mv_path = pjoin(data_dir, "MotionVector.{:04d}.exr".format(idx))

        
            img_view = load_exr(low_img_path)
            img_view_truth = load_exr(high_img_path)
            img_depth = load_exr(depth_img_path, 1)
            img_flow = load_exr(mv_path, 2)

            img_flow = upsampling_mv(img_flow)

            trans = self.transform

            img_view_truth = trans(img_view_truth)
            img_flow = trans(img_flow)

            img_view = trans(img_view)
            # depth data is in a single-channel image.
            img_depth = trans(img_depth)

            if self.cropped_size is not None:

                if crop_l is None:

                    crop_l, crop_r, crop_t, crop_b = [], [], [], []

                    for i in range(self.cropped_num):

                        crop_l.append(torch.randint(0, img_view.shape[2] - self.cropped_size[1], ()).item())
                        crop_r.append(crop_l[i] + self.cropped_size[1])
                        crop_t.append(torch.randint(0, img_view.shape[1] - self.cropped_size[0], ()).item())
                        crop_b.append(crop_t[i] + self.cropped_size[0])

                    # crop_l = torch.randint(0, img_view.shape[1] - self.cropped_size[1], ()).item()
                    # crop_r = crop_l + self.cropped_size[1]
                    # crop_t = torch.randint(0, img_view.shape[0] - self.cropped_size[0], ()).item()
                    # crop_b = crop_t + self.cropped_size[0]
                
                img_view_list = []
                img_view_truth_list = []
                img_depth_list = []
                img_flow_list = []
                for i in range(self.cropped_num):
                    
                    l, r, t, b = crop_l[i], crop_r[i], crop_t[i], crop_b[i]

                    img_view_list.append(img_view[:, t:b, l:r])
                    img_view_truth_list.append(img_view_truth[:, t*2:b*2, l*2:r*2])
                    img_depth_list.append(img_depth[:, t:b, l:r])
                    img_flow_list.append(img_flow[:, t*2:b*2, l*2:r*2])

                img_view = torch.stack(img_view_list, dim=0)
                img_view_truth = torch.stack(img_view_truth_list, dim=0)
                img_depth = torch.stack(img_depth_list, dim=0)
                img_flow = torch.stack(img_flow_list, dim=0)

                # img_view = img_view[crop_t:crop_b, crop_l:crop_r, :]
                # img_view_truth = img_view_truth[crop_t*2:crop_b*2, crop_l*2:crop_r*2, :]
                # img_depth = img_depth[crop_t:crop_b, crop_l:crop_r]
                # img_flow = img_flow[crop_t:crop_b, crop_l:crop_r, :]


            
            view_list.append(img_view)
            depth_list.append(img_depth)
            flow_list.append(img_flow)
            truth_list.append(img_view_truth)
            
        return view_list, depth_list, flow_list, truth_list[0]

    def __len__(self) -> int:
        return len(self.data_list)

