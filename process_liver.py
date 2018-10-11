# -*- coding:utf-8 -*-
'''
process the dataset to liver and lesion 
'''

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
import scipy.io as sio

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


niftis_path = '/media/lab150/Data/LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 1/'        
root_process_database = '/media/lab150/Data/LiTS/LiTS_database/'
folder_volumes = root_process_database + 'images_volumes/'
folder_seg_livers = root_process_database + 'liver_seg/'
folder_seg_items = root_process_database + 'item_seg/'

mkdir_if_not_exist(root_process_database)
mkdir_if_not_exist(folder_volumes)
mkdir_if_not_exist(folder_seg_livers)
mkdir_if_not_exist(folder_seg_items)

files_path = glob.glob(niftis_path+"*.nii")
filenames = []
list_file_names = []

for file_path in files_path:
    if (os.path.basename(file_path)[0] == 'v') or (os.path.basename(file_path)[0] == 's'):
        filenames.append(file_path)
        
for filename in filenames:
    if os.path.basename(filename)[0] == 'v':
        print('Processing Volume')
        name = os.path.basename(filename).replace('.nii','').replace('volume-','')
        folder_volume = folder_volumes+name
        mkdir_if_not_exist(folder_volume)
        
        volume = nib.load(filename)
        imgs = volume.get_data()
        imgs[imgs<-150] = -150
        imgs[imgs>250] = 250
        img_volume = 255*(imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
        
        for i in range(img_volume.shape[2]):
            section = img_volume[:,:,i]
            filename_for_section = os.path.join(folder_volume, '{}.mat'.format(i+1))
            sio.savemat(filename_for_section, {'section': section})
    
    elif os.path.basename(filename)[0] == 's':
        print('Processing Segmentation')
        name = os.path.basename(filename).replace('.nii','').replace('segmentation-','')
        folder_seg_item = folder_seg_items+name
        mkdir_if_not_exist(folder_seg_item) 
        folder_seg_liver = folder_seg_livers+name
        mkdir_if_not_exist(folder_seg_liver) 
         
        segmentation = nib.load(filename) 
        img_seg = segmentation.get_data()
        img_seg_item = img_seg.copy()
        img_seg_liver = img_seg.copy()
        img_seg_item[img_seg_item == 1]=0
        img_seg_item[img_seg_item == 2]=1
        img_seg_liver[img_seg_liver == 2]=1
        
        for i in range(img_seg_item.shape[2]):
            item_seg_section = img_seg_item[:,:,i]*255
            liver_seg_section = img_seg_liver[:,:,i]*255
            
            filename_for_seg_item_section = os.path.join(folder_seg_item, '{}.png'.format(i+1))
            filename_for_seg_liver_section = os.path.join(folder_seg_liver, '{}.png'.format(i+1))
            cv2.imwrite(filename_for_seg_item_section,item_seg_section)
            cv2.imwrite(filename_for_seg_liver_section,liver_seg_section)