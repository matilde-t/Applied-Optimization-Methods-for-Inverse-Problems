import aomip
import matplotlib.pyplot as plt
import tifffile
import os

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

    

if __name__ == '__main__':
    printShell()
    # test_Slicing()