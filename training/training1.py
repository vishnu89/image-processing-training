from skimage import img_as_ubyte
from skimage.color import rgb2gray

from common_utils import *
from matplotlib import pyplot as plt
# from projectConfig import *
# sample_freq = 220j
# order = 2
# x, y = np.mgrid[-1:1:sample_freq, -1:1:sample_freq]
# img4 = 1 * np.exp(-2*(x**order+y**order))
# imshow("img4  6 training1", img4, 0)
# stitch_img()
# plt.subplots()
# img1 = cv2.imread(r'/home/dondana/vishnu/workspace_py/dataBase/training/lena.jpg')
# imshow("img1  10 training1", img1, 0)
# imshow("img1  10 training1", img1[::1,::10], 0)
#########################

# a = cv2.imread(r'/home/dondana/vishnu/workspace_py/dataBase/training/sampling.jpg',0)
# img1 = resize(img1,(10,10))
# imshow("img1  17 training1", a, 1)
#
# b = a.copy()
# n = 2
# step = np.linspace(0,256,n+1).astype('u2')
# print step
# nstep = step[1:]
# for ps, ns in zip(step, nstep):
#     print ns-ps
#     a[(ps<=a)&(a<=ns)] = ps
# imshow("img1  10 training1", a.astype('u1'), 0)


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
        ax.imshow(img,**kw)
    plt.show()


img1 = plt.imread(r'/home/dondana/vishnu/workspace_py/dataBase/training/gumby.bmp')
img1 = img_as_ubyte(rgb2gray(img1))

def edge_detection(scale =1):
    img2 = img1.copy()
    img3 = img1.copy()
    img4 = np.zeros_like(img1).copy()

    img2[:-scale] = img1[scale:]
    img2 = img1 - img2
    img3[scale:] = img1[:-scale]
    img3 = img1 - img3
    img4[img2 == 255] = 255
    img4[img3 == 255] = 255
    imshow(img1, img2, img3, img4, cmap='binary')


edge_detection()

'''
# The convolution operation is applicable for linear time invariant (LTI) systems only
x = img1.astype('u8') #np.array([-1,-1,-1,1,1,1])
y = img2.astype('u8') #np.array([1,1,1,1,1,1])
sxx = x*x.T - x.sum()*x.sum()/x.size
syy = y*y.T - y.sum()*y.sum()/y.size
sxy = x*y.T - x.sum()*y.sum()/x.size
rho = sxy / (sxx*syy)**.5
print sxx
print syy
print sxy
print rho
'''

'''
http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d
for(i=0; i < rows; ++i)              // rows
{
    for(j=0; j < cols; ++j)          // columns
    {
        for(m=0; m < kRows; ++m)     // kernel rows
        {
            mm = kRows - 1 - m;      // row index of flipped kernel

            for(quantize_level=0; quantize_level < kCols; ++quantize_level) // kernel columns
            {
                nn = kCols - 1 - quantize_level;  // column index of flipped kernel

                // index of input signal, used for checking boundary
                ii = i + (m - kCenterY);
                jj = j + (quantize_level - kCenterX);

                // ignore input samples which are out of bound
                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
                    out[i][j] += in[ii][jj] * kernel[mm][nn];
            }
        }
    }
}
'''

x = np.array([1, 1, 1, 1, 0, 0, 0, 0])
edge = x[:-1] - x[1:]
edge[edge <= 0] = 0  # quantization
edge[edge > 0] = 1  # quantization
print edge
#
x = np.array([1, 1, 1, 1, 0, 0, 0, 0])
edge = x[1:] - x[:-1]
edge[edge <= 0] = 0 # quantization
edge[edge > 0] = 1 # quantization
print edge
#
# h1 = np.array([1,-1])
# h2 = np.array([-1,1])
#

def corr2(a, k):
    a = a.astype('i8')
    k = k.astype('i8')
    out = np.zeros_like(a)
    rows, cols = a.shape
    krows, kcols = k.shape
    print rows, cols, krows, kcols
    # find center position of kernel (half of kernel size)
    kxcenter = kcols / 2
    kycenter = krows / 2
    for row in range(rows):
        for col in range(cols):
            for krow in range(krows):
                flip_row = krows - 1 - krow
                for kcol in range(kcols):
                    flip_col = kcols - 1 - kcol
                    irow = row + (krow - kycenter)
                    icol = col + (kcol - kxcenter)
                    if 0 <= irow < rows and 0 <= icol < cols:
                        out[row][col] += a[irow][icol] * k[flip_row][flip_col]
    return out

# img1 = np.arange(1, 10).reshape(3, -1)


pix = 1
min,max = 1,-1
chess_board = np.ones([8, 8, pix, pix], 'i8')*min
mark = np.ones([pix, pix], 'i8') * max
chess_board[::2,::2] = mark
chess_board[1::2,1::2] = mark
chess_board = np.hstack(np.hstack(chess_board))

print chess_board.shape
# imshow("chess_board  117 training1", chess_board, 0)



# img1 = chess_board
# img1 = np.arange(1, 5).reshape(2, 2)
b = np.array([[1, -1]])
# print img1.shape

val = corr2(chess_board, b)
print val
val[val<=0] = 0
val[val>1] = 255
imshow("  155 training1", val.astype('u1'), 0)
