from aomip import OGM1, GD , LW
from challenge.utils import load_htc2022data, segment, calculate_score
import matplotlib.pyplot as plt
import numpy as np
import tifffile

def test_OGM1():
    sino, A = load_htc2022data('/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full')
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread('/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(sino, cmap='gray')
    ax[0].set_title('Sinogram')
    ax[1].imshow(ground, cmap='gray')
    ax[1].set_title('Ground truth')
    fig.suptitle('Helsinki original data')
    plt.tight_layout()
    plt.savefig('./homework/hw03/htc2022_orig.png')
    
    gd = GD(A, sino, x0)
    ogm = OGM1(A, sino, x0)

    res=[]
    res.append(gd.leastSquares())
    res.append(gd.l2Norm())
    res.append(ogm.leastSquares())
    res.append(ogm.l2Norm())

    titles = ['GD-LeastSquares', 'GD-Tikhonov', 'OGM1-LeastSquares', 'OGM1-Tikhonov']
    score = []
    x = np.arange(2, 6)
    fig, ax = plt.subplots(2, 2)
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].imshow(res[i*2+j], cmap='gray')
            ax[i, j].set_title(titles[i*2+j])
            tifffile.imwrite('./homework/hw03/htc2022_'+titles[i*2+j]+'.tif', res[i*2+j])
            score.append(calculate_score(segment(res[i*2+j]), segment(ground)))
    fig.suptitle('Comparison of different methods')
    plt.tight_layout()
    plt.savefig('./homework/hw03/GD_OGM.png')

    plt.figure()
    plt.scatter(x, score)
    plt.xticks(x, titles)
    plt.ylim(0, 1)
    plt.title('Score of different methods')
    plt.savefig('./homework/hw03/GD_OGM_err.png')

    print(score)

    return

def test_Landweber():
    sino, A = load_htc2022data('/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full')
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)

    ground = tifffile.imread('/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif')

    lw = LW(A, sino, x0)
    sigma2 = lw.sigma2()

    mult = [1, 1e2, 1e4, 1e6]
    lam = [1/(sigma2*i) for i in mult]
    res = []
    score = []
    for l in lam:
        img = lw.solve(l)
        res.append(img)
        score.append(calculate_score(segment(img), segment(ground)))

    fig, ax = plt.subplots(2, 2)
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].imshow(res[i*2+j], cmap='gray')
            ax[i, j].set_title('lambda=1/(sigma^2*'+str(int(mult[i*2+j]))+')')
            tifffile.imwrite('./homework/hw03/htc2022_Landweber_'+str(mult[i*2+j])+'.tif', res[i*2+j])
    fig.suptitle('Landweber iteration')
    plt.tight_layout()
    plt.savefig('./homework/hw03/landweber.png')

    plt.figure()
    plt.scatter(lam, score)
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.title('Score of different lambda')
    plt.tight_layout()
    plt.savefig('./homework/hw03/landweber_score.png')

    print(score)

    return

def test_all():
    test_OGM1()
    test_Landweber()
    return

if __name__ == '__main__':
    sino, A = load_htc2022data('/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full')
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)
    lw = LW(A, sino, x0)
    img = lw.solve()
    tifffile.imwrite('./homework/hw03/htc2022_Landweber.tif', img)