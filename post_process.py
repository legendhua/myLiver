# -*- coding: utf-8 -*-

'''
分割预测结果的后处理
'''

import skimage.measure as measure
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import numpy as np
import cv2
import matplotlib.pyplot as plt

def post_process(pred_seg): 
    # 边界处理
    pred_seg = clear_border(pred_seg) 
    # 空洞填充  
    edges = roberts(pred_seg)
    pred_seg = ndi.binary_fill_holes(edges)
    pred_seg = pred_seg.astype(np.uint8)
    plt.imshow(pred_seg,cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    data = cv2.imread('116.png',0)/255
    post_process(data)