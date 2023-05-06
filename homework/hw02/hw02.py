import aomip
import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np


def test_Slicing():
    path = '/srv/ceph/share-all/aomip/6983008_seashell'
    img = []
    for image in os.listdir(path):
        tmp = aomip.centerOfRotationCorrection(tifffile.imread(os.path.join(path, image)), -4, 1) # correction as suggested in the dataset
        img.append(tmp)
    sino = []
    idx = []
    for i in range(0,2240,500):
        sino.append(aomip.slice(img, i))
        idx.append(i)
    n = len(sino)
    fig, ax = plt.subplots(1,n-1)
    for i in range(1,n):
        ax[i-1].imshow(sino[i], cmap='gray')
        ax[i-1].set_title('Row {}'.format(idx[i]))
    fig.suptitle('Sinograms of different rows')
    plt.tight_layout()
    plt.savefig('./homework/hw02/sinogram.png')
    return

def printShell():
    img = tifffile.imread('/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif')
    img = aomip.centerOfRotationCorrection(img, -4, 1)
    img = aomip.binning(img, 3)
    plt.imsave('./homework/hw02/seashell.png', img, cmap='gray')
    return

def test_GradientDescent():
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    x0 = np.array([-5, -5])
    l_val = [1, 1e-1, 1e-2, 1e-3, 1e-4]

    # booth = lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    # dbooth = lambda x: np.array([10 * x[0] + 8 * x[1] - 34, 
    #                              8 * x[0] + 10 * x[1] - 38])
    # Z = booth([X, Y])
    # plt.figure()
    # plt.contour(X, Y, Z, 1000)
    # plt.colorbar()
    # plt.scatter(1, 3, c='r')
    # plt.title('Booth function')
    # plt.savefig('./homework/hw02/booth.png')
    # print('Booth function')
    # res = {}
    # for l in l_val:
    #     x = aomip.gradDesc(dbooth, x0, l)
    #     res[l] = x
    # print(res)

    camel = lambda x: 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2
    dcamel = lambda x: np.array([4 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1],
                                    x[0] + 2 * x[1]])
    Z = camel([X, Y])
    plt.figure()
    plt.contour(X, Y, Z, 1000)
    plt.colorbar()
    plt.scatter(0, 0, c='r')
    plt.title('Three-hump camel function')
    plt.savefig('./homework/hw02/camel.png')
    print('Camel function')
    res = {}
    for l in l_val:
        x = aomip.gradDesc(dcamel, x0, l, 1e7)
        res[l] = x
    print(res)
        

    

if __name__ == '__main__':
    # printShell()
    # test_Slicing()
    test_GradientDescent()