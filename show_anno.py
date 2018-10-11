import json
import numpy as np
from matplotlib import pyplot as plt

with open('IM000148.anno') as file:
    x_y_list = file.readline()
    Nodule_Attributions_Dict = json.loads(x_y_list)
    Coords = Nodule_Attributions_Dict['Coords']
    CoordsX = Coords[0::2]
    CoordsY = Coords[1::2]
    anno = np.zeros((512,512))
    anno[CoordsY,CoordsX] = 255
    plt.imshow(anno,cmap='gray')
    plt.show()