# -*- coding: utf-8 -*-
'''
标注anno文件的处理：
包括消除完整标注框、消除不完整标注框、肝部轮廓不连续填充等
'''
import numpy as np
import cv2
import json
import skimage
from skimage.filters import roberts
from scipy import ndimage as ndi

# 删除完整矩形连续框
def delete_rectangle(img):
    # 完整边框像素大小，可修改
    pixel_rect = 5
    # 原始标注图像
    cc = cv2.connectedComponents(1-img,connectivity=4)
    #print(cc[0])
    for i in range(0,cc[0]-1):
    # -- judge the rectangle area
        ccmap = (cc[1]==(i+1)).astype(np.uint8) # 0黑色为内部轮廓边界线连通域，不可能构成矩形
        x,y,w,h = cv2.boundingRect(ccmap)
        if ccmap[y:y+h,x:x+w].all():
            img[y-pixel_rect:y+h+pixel_rect,x-pixel_rect:x+w+pixel_rect] = 0
    return img
    

def anno_process(anno_data):
    '''
    输入：anno标注数据，二维矩阵
    '''
    anno_data = np.array(anno_data,np.uint8)
    # 如果数据标注非常小，则将不进行处理
    # 原始标注图像
    cv2.imwrite('anno.png',anno_data*255)
    if (np.sum(anno_data))<10:
        return anno_data
    # step1: 连通域标准矩形法消除完整标注框，原始标注存在连续矩形框
    anno_data = delete_rectangle(anno_data)
    
    # step2: 动态膨胀腐蚀法消除不连续边框，将不连续轮廓处理为连续轮廓
    iter = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    max_iter = 5 # 最大迭代次数
    cc = cv2.connectedComponents(1-anno_data,connectivity=4)
    last_num_of_connect_map = cc[0]
    print(cc[0])
    while cc[0] < 3:
        anno_data = cv2.dilate(anno_data,kernel,iterations = 1)
        #膨胀后可能会出现连续矩形框，所以需要删除
        anno_data = delete_rectangle(anno_data) 
        cc = cv2.connectedComponents(1-anno_data,connectivity=4)
        print(cc[0])
        map = show_connect_map(cc[1],cc[0])
        cv2.imwrite('cmap.png',map)
        iter += 1
        if iter > max_iter:
            break
    anno_data = cv2.dilate(anno_data,kernel,iterations = 1)# 因为可能存在局部小区域的连通域或者已经闭合的轮廓但是存在矩形框线
    edges = roberts(anno_data)
    anno_data = ndi.binary_fill_holes(edges).astype(np.uint8)
    # 腐蚀操作
    if iter >=0:
        anno_data = cv2.erode(anno_data,kernel,iterations = iter+2)
    # 处理后的标注图像
    cv2.imwrite('anno_result.png',anno_data*255)
    return anno_data


def show_connect_map(cmap,num_connect):
    num_gray = 255 // (num_connect-1)
    for i in range(num_connect):
        cmap[np.where(cmap == i)] = num_gray * i
    return cmap

if __name__ == '__main__':
    annopath = 'IM000140.anno'
    with open(annopath) as file:
        x_y_list = file.readline()
        Nodule_Attributions_Dict = json.loads(x_y_list)
        Coords = Nodule_Attributions_Dict['Coords']
        CoordsX = Coords[0::2]
        CoordsY = Coords[1::2]
        try:
            height = Nodule_Attributions_Dict['annoImgHeight']
        except KeyError:
            print(annopath+'没有annoImgHeight参数')
            height = 1024
        anno = np.zeros((height,height))
        # 轮廓
        if max(CoordsX) < height and max(CoordsY) < height:
            anno[CoordsX,CoordsY] = 1
            print('procing'+annopath)
            anno_process(anno)
