import numpy as np
import cv2
import json
import os
import csv
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# params={
#     'axes.labelsize': '5',
#     'xtick.labelsize':'15',
#     'ytick.labelsize':'15',
#     'lines.linewidth':0.75,
#     'legend.fontsize': '10',
#     'figure.figsize'   : '12, 9'    # set figure size
# }
# pylab.rcParams.update(params)

# with open('contour.txt') as file:
#     x_y_list = file.readlines()
#     x_length = len('    <edgeMap><xCoord>')
#     y_length = x_length + len('</xCoord><yCoord>') + 3
#     x = [item_x[x_length : x_length + 3] for item_x in x_y_list]
#     y = [item_y[y_length : y_length + 3] for item_y in x_y_list]
# fig = plt.figure(1,dpi=50)
# plt.plot(x , y, linewidth = 0.75)
# plt.savefig('contour.png', dpi=100)
# plt.show()
# im = test[0].reshape(128,128)
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im,cmap='gray')
# plt.show()

def writeAnnosCSV(filename,annos):
    ##生成seriesuids.csv用于FROC评估。
    firstline = ['seriesuid','coordX', 'coordY', 'coordZ','diameter_mm']
    #with open(filename, 'wt', newline='') as f:#Python3.6，若不指定newline=''则会每写一行后写一个空行。
    with open(filename, 'wt') as f:#Python2.7不用指定newline=''。
        fwriter = csv.writer(f)
        fwriter.writerow(firstline)
        for row in annos:
            fwriter.writerow(row)


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

annos = []
for anno_item in os.listdir('./annos'):
    if anno_item.endswith('.anno'):
        with open('./annos/'+ anno_item) as file:
            try:
                x_y_list = file.readline()
                Nodule_Attributions_Dict = json.loads(x_y_list)
                Origin = Nodule_Attributions_Dict['Origin']
                Spacing = Nodule_Attributions_Dict['Spacing']
                Coords = Nodule_Attributions_Dict['Coords']
                Dimension = Nodule_Attributions_Dict['Dimension']
                SeriesID = Nodule_Attributions_Dict['SeriesID']
                InstanceNumber = Nodule_Attributions_Dict['InstanceNumber']
                CoordsXY = np.array(Coords)
                CoordsX_p = CoordsXY[::2]
                CoordsY_p = CoordsXY[1::2]
                CoordsXY_list = np.dstack((CoordsX_p, CoordsY_p))
                CoordsXY_list2 = []
                for item in CoordsXY_list[0]:
                    CoordsXY_list2.append([list(item)])
                CoordsXY_list2 = np.array(CoordsXY_list2)
                (cx, cy), radius = cv2.minEnclosingCircle(CoordsXY_list2)#最小外接圆
                center = (int(cx),int(cy))
                x_WorldCoord, y_WorldCoord = VoxelToWorldCoord(np.array([cx, cy]),np.array(Origin[0:2]),np.array(Spacing[0:2]))
                z_WorldCoord = Origin[2] + (Dimension[2] - InstanceNumber)*Spacing[2]
                Diameter_mm = 2*radius*Spacing[0]
                annos.append([SeriesID, x_WorldCoord, y_WorldCoord, z_WorldCoord, Diameter_mm])
                # radius = int(radius)
                # img = np.ones((512,512,3), np.uint8)*255
                # cv2.drawContours(img, CoordsXY_list2,-1,(0,0,255),3)
                # cv2.circle(img,center,1,(0,0,0),2)
                # cv2.circle(img,center,radius,(255,0,0),2)
                # cv2.imwrite('contour_{}.png'.format(anno_item[:-5]),img)
            except:
                pass
writeAnnosCSV('test_anno.csv', annos)