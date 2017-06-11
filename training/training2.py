from common_utils import *
# from projectConfig import *
from matplotlib import pyplot as plt

img = cv2.imread(r'/home/dondana/vishnu/workspace_py/dataBase/training/lena.jpg',0)


def imshow(*imgs, **kw):
    nimg = len(imgs)
    if kw.get('col') is None:
        col = nimg ** .5
        col = int(col + 1 if col % 1 else col)
    else:
        col = kw.pop('col')
    row = nimg / col
    row = row + 1 if (col * row) - nimg else row
    if row == col == 1:
        plts = [plt.subplots(row,col,figsize=kw.pop('figsize',(10,10)))[1]]
    else:
        plts = plt.subplots(row,col,figsize=kw.pop('figsize',(10,10)))[1].ravel()
    for img, ax in zip(imgs, plts):
        if len(img)==2:
            name, img = img
            ax.imshow(img,**kw)
            ax.set_title(name)
        else:
            ax.imshow(img,**kw)
    plt.show()




