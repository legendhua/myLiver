#-*- coding:utf-8 -*-
'''
函数功能：nifti数据的处理
'''
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

def test():
    data = nib.load('data/volume-112.nii')
    seg = nib.load('data/segmentation-112.nii')

    print("data shape:",data.shape)

    print("seg shape:",seg.shape)
    f1 = plt.figure()
    plt.imshow(data.get_data()[:,:,56].T,cmap="gray")
    plt.show()
    f2 = plt.figure()
    plt.imshow(seg.get_data()[:,:,56].T,cmap="gray")
    plt.show()
    f3 = plt.figure()
    plt.hist(seg.get_data()[:,:,56].flatten(),bins=200)
    plt.show()
    print(np.unique(seg.get_data()[:,:,56]))
    print(np.where(seg.get_data()[:,:,56]==1)[0].shape)
    print(np.where(seg.get_data()[:,:,56]==2)[0].shape)

def show_liver(data_path, seg_path):
    data = nib.load(data_path)
    img = data.get_data()
    seg = nib.load(seg_path)
    seg_data = seg.get_data()
    store_path = '../results/ground truth'
    img_path = '../results/CT_scan'
    if not os.path.exists(store_path): os.makedirs(store_path)
    if not os.path.exists(img_path): os.makedirs(img_path)
    
    for i in range(img.shape[2]):
        save_path = os.path.join(store_path, 'GT-{}.png'.format(i+1))
        cv2.imwrite(save_path, seg_data[:,:,i]*255)
        save_path = os.path.join(img_path, 'CT-{}.png'.format(i+1))
        cv2.imwrite(save_path, img[:,:,i])

if __name__ == '__main__':
    segPath = '/media/lab150/Data/LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 2/segmentation-110.nii'
    imgPath = '/media/lab150/Data/LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 2/volume-110.nii'
    show_liver(imgPath, segPath)
    print("finished")
