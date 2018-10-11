# -*- coding:utf-8 -*-
'''
处理自有肝数据，数据格式为由dcm转为的nii数据
'''
import glob
import os
import cv2
import json
import SimpleITK as sitk
import numpy as np
import scipy.io as sio
import skimage
from skimage.filters import roberts
from scipy import ndimage as ndi
from setuptools.sandbox import save_path

def dice_sh(labelFiles,predictFiles):
    labelfiles = glob.glob(labelFiles+'/*.png')
    predictfiles = glob.glob(predictFiles+'/*.png')
    num = len(labelfiles)
    total_dice = 0.0
    for n in range(num):
        labelData = cv2.imread(labelfiles[n],0)
        predictData = cv2.imread(predictfiles[n],0)
        dice = dice_coef_theoretical(labelData, predictData)
        total_dice += dice
    print(total_dice/num)
         
   
def dice_coef_theoretical(y_true, y_pred):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
    """
    y_true = np.where(y_true==255,1,0).astype(np.float32)
    y_pred = np.where(y_pred==255,1,0).astype(np.float32)

    y_true_f = np.reshape(y_true, [-1])

    y_pred_f = np.reshape(y_pred, [-1])

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00000001)

    if (np.sum(y_pred) == 0) and (np.sum(y_true) == 0):
        dice = 1

    return dice

def process_data(filename):
    folder_volume = os.path.join(os.path.dirname(os.path.dirname(filename)),'volume')
    if not os.path.exists(folder_volume): os.makedirs(folder_volume)
    volume = nib.load(filename) 
    imgs = volume.get_data()
    imgs[imgs<-150] = -150
    imgs[imgs>250] = 250
    img_volume = 255*(imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
    
    for i in range(img_volume.shape[2]):
        section = img_volume[:,:,i]
        filename_for_section = os.path.join(folder_volume, '{}.mat'.format(i+1))
        sio.savemat(filename_for_section, {'section': section})

def process_label(filepath):
    # 将真实标注转化为需要的.png格式
    num = 1
    folder_seg = os.path.join(os.path.dirname(filepath),'liver_seg')
    if not os.path.exists(folder_seg): os.makedirs(folder_seg)
    for file in sorted(os.listdir(filepath),reverse=True):
        file_for_slice_seg = os.path.join(folder_seg,'{}.png'.format(num))
        label = read_anno(os.path.join(filepath,file))
        cv2.imwrite(file_for_slice_seg,label*255)
        num += 1
        

def read_anno(annoname):
    with open(annoname) as file:
        x_y_list = file.readline()
        Nodule_Attributions_Dict = json.loads(x_y_list)
        Coords = Nodule_Attributions_Dict['Coords']
        CoordsX = Coords[0::2]
        CoordsY = Coords[1::2]
        anno = np.zeros((512,512))
        if max(CoordsX)>512 or max(CoordsY)>512:
            print(annoname)
        else:
            anno[CoordsX,CoordsY] = 1
            edges = roberts(anno)
            anno = ndi.binary_fill_holes(edges)
        

    return np.rot90(anno).astype(np.int16)

# 将上海给定的肝标注数据转化为Nii格式
def save_nii(ct_path, label_path):
    # 读取ct数据，用于求分辨率和原点
    ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
    ct_data = sitk.GetArrayFromImage(ct)
    print('the shape of ct:',ct_data.shape)
    name = os.path.basename(ct_path).split('.')[0].replace('volume','segmentation')
    # 将标注文件处理成矩阵格式
    label = [] 
    for file in sorted(glob.glob(label_path+'/*.anno'),reverse=True):
        label_slice = read_anno(file)
        label.append(label_slice)
    save_path = os.path.join(os.path.dirname(os.path.dirname(ct_path)),'seg')
    if not os.path.exists(save_path): os.makedirs(save_path)
    label = np.array(label)
    print('the shape of label:',label.shape)
    label = sitk.GetImageFromArray(label)
    label.SetDirection(ct.GetDirection())
    label.SetOrigin(ct.GetOrigin())
    label.SetSpacing(ct.GetSpacing())
    sitk.WriteImage(label, os.path.join(save_path,name+'.nii'))

def generate_txt():
    basedir = 'ZS15292934/volume/'
    num = 300
    with open('ZS15292934_sh.txt','w') as w:
        for i in range(1,num-1,1):
            line = basedir+str(i)+'.mat '+basedir+str(i+1)+'.mat '+basedir+str(i+2)+'.mat\n'
            w.write(line)
        
if __name__ == '__main__':
    #filepath = '../LiTS_database/liver_data_sh/ZS15292934/SE01/20151203_07044861USERPTCTET3Ds003a001.nii'
    #process_data(filepath)
    #filepath = '../LiTS_database/liver_data_sh/ZS15292934/anno'
    #process_label(filepath)
    #generate_txt()

    #labelPath = r'F:\my_learning\DL\research\Medical\liver_and_tumor_seg\data\ZS15127160\label_image'
    #predictPath = r'F:\my_learning\DL\research\Medical\liver_and_tumor_seg\data\ZS15127160\test_sh'
    #dice_sh(labelPath, predictPath)
    
    ct_nii_path = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/VNet_data/sh_data/ct'
    seg_anno_path = '/media/omnisky/5207f9eb-00ef-4e6f-8d90-1cd7f75edb86/zgh/liver_sh/'
    for nii_file in os.listdir(ct_nii_path):
        patient_num = nii_file.split('.')[0].split('_')[0]
        ct_path = os.path.join(ct_nii_path,nii_file)
        anno_path = os.path.join(seg_anno_path,patient_num,'SE02')
        save_nii(ct_path,anno_path)
