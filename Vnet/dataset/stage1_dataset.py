# -*- coding:utf-8 -*-
"""

固定取样方式下的数据集
"""

import os

import torch
import SimpleITK as sitk
from torch.utils.data import Dataset as dataset

on_server = True


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


# 第一阶段的训练数据
train_ct_dir = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/train/ct' \
    if on_server is False else '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/train/ct'
train_seg_dir = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/train/seg' \
    if on_server is False else '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/train/seg'

train_fix_ds = Dataset(train_ct_dir, train_seg_dir)

# 验证数据
val_ct_dir = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/test/ct'
val_seg_dir = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/test/seg'

val_fix_ds = Dataset(val_ct_dir, val_seg_dir)

# # 娴嬭瘯浠ｇ爜
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from torch.utils.data import DataLoader
#
train_dl = DataLoader(train_fix_ds, 2, True, num_workers=2, pin_memory=True)
for index, (ct, seg) in enumerate(train_dl):
    # print(type(ct), type(seg))
    print(index, ct.size(), seg.size())
    print('----------------')
'''
