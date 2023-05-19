from aomip import OGM1, GD , iradon, XrayOperator
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def testOGM1():
    data = loadmat('/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_01c_full', simplify_cells=True, squeeze_me=True)
    sino = data['CtDataFull']['sinogram'].T
    angles = data['CtDataFull']['parameters']['angles']
    source_origin = data['CtDataFull']['parameters']['distanceSourceOrigin']*100
    source_detector = data['CtDataFull']['parameters']['distanceSourceDetector']
    x_shape = np.array([512, 512])

    x0 = iradon(sino, [sino.shape[0]], x_shape, angles, source_origin, source_detector)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(sino, cmap='gray')
    ax[0].set_title('Sinogram')
    ax[1].imshow(np.asarray(Image.open('/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_01c_recon_fbp.png')), cmap='gray')
    ax[1].set_title('Reference image')
    ax[2].imshow(x0, cmap='gray')
    ax[2].set_title('Reconstructed image')
    fig.suptitle('Helsinki original data')
    plt.tight_layout()
    plt.savefig('./homework/hw03/htc2022_orig.png')

    A = XrayOperator(
        x_shape,
        [sino.shape[0]],
        angles,
        source_origin,
        source_detector)
    
    gd = GD(A, sino, x0)
    ogm = OGM1(A, sino, x0, l=1e-4)

    res=[]
    res.append(gd.leastSquares())
    res.append(gd.l2Norm())
    res.append(ogm.leastSquares())
    res.append(ogm.l2Norm())

    titles = ['GD Least squares', 'GD Tikhonov', 'OGM1 LS', 'OGM1 Tikhonov']
    fig, ax = plt.subplots(2, 2)
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].imshow(res[i*2+j], cmap='gray')
            ax[i, j].set_title(titles[i*2+j])
    fig.suptitle('Comparison of different methods')
    plt.tight_layout()
    plt.savefig('./homework/hw03/htc2022_comp.png')

if __name__ == '__main__':
    testOGM1()