#-*- coding:utf-8 -*-
'''
函数功能：nifti数据的处理
'''
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom
import gzip
import SimpleITK as sitk

def test_nii():
    '''
    some simple tests on liver data 
    '''
    sitkImage = sitk.ReadImage(r'H:\202_xk_chest_c-test_thorax.nii.gz')
    # 原始数据
    volume = sitk.GetArrayFromImage(sitkImage)
    print(volume.shape)
    # 分辨率
    spacing = sitkImage.GetSpacing()
    print(spacing)
    # 原点
    origin = sitkImage.GetOrigin()
    print(origin)
    
    #image = nib.load('data/ZS14264487.nii').get_data()[:,:,50]
    #plt.imshow(volume[...,250].T,cmap="gray")
    #plt.show()
#     print(np.max(image))
#     print(np.min(image))
#     plt.hist(image.flatten(),bins=200)
#     plt.show()
    
#     data = nib.load('data/volume-112.nii')
#     seg = nib.load('data/segmentation-112.nii')
#     image = np.array(data.get_data()[:,:,40])
#     print(np.max(image))
#     print(np.min(image))
#     plt.hist(image.flatten(),bins=200)
#     plt.show()
    
#     print("data shape:",data.shape)
# 
#     print("seg shape:",seg.shape)
#     f1 = plt.figure()
#     plt.imshow(data.get_data()[:,:,56].T,cmap="gray")
#     plt.show()
#     f2 = plt.figure()
#     plt.imshow(seg.get_data()[:,:,56].T,cmap="gray")
#     plt.show()
#     f3 = plt.figure()
#     plt.hist(seg.get_data()[:,:,56].flatten(),bins=200)
#     plt.show()
#     print(np.unique(seg.get_data()[:,:,56]))
#     print(np.where(seg.get_data()[:,:,56]==1)[0].shape)
#     print(np.where(seg.get_data()[:,:,56]==2)[0].shape)

def test_dcm():
    liver = pydicom.read_file("data/IM000026")
    image = np.array(liver.pixel_array.astype(np.int16))
    print(np.max(image))
    print(np.min(image))
    plt.hist(image.flatten(),bins=200)
    plt.show()

def show_liver(data_path, seg_path):
    '''
    show the CT data and the liver seg data
    ---parameters---
    data_path: CT data path
    seg_path； seg data path
    ---return---
    '''
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

def nii2image(filepath, savepath):
    # 将Nii格式的数据转化为图片的形式，其中数据的大小需要考虑是否乘以255
    if not os.path.exists(savepath): os.makedirs(savepath)
    sitkImage = sitk.ReadImage(filepath)
    image = sitk.GetArrayFromImage(sitkImage)

    for i in range(image.shape[0]):
        cv2.imwrite(os.path.join(savepath,'seg-{}.png'.format(i)), image[i,:,:]*255)
        
if __name__ == '__main__':
#     segPath = '/media/lab150/Data/LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 2/segmentation-110.nii'
#     imgPath = '/media/lab150/Data/LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 2/volume-110.nii'
#     show_liver(imgPath, segPath)
#     print("finished")    
    test_nii()
    
#     niifile = r'./data/ZS15292934_segmentation.nii'
#     savepath = r'./data/prediction'
#     nii2image(niifile, savepath)
    