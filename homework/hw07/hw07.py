from aomip import ADMM
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def smooth(N):
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    sigma = 0.25 * N
    c = np.array([[0.6*N, 0.6*N], [0.5*N, 0.3*N], [0.2*N, 0.7*N], [0.8*N, 0.2*N]])
    a = np.array([1, 0.5, 0.7, 0.9])
    img = np.zeros((N, N))
    for i in range(4):
        term1 = (I - c[i, 0])**2 / (1.2 * sigma )**2
        term2 = (J - c[i, 1])**2 / sigma**2
        img += a[i] * np.exp(-term1 - term2)
    return img

def test_ADMM_LASSO():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full", 60
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    nmax = 10

    admm = ADMM(A, sino, x0, nmax=nmax)

    for tau in np.logspace(-3, 6, 10):
        img = admm.LASSO(tau=tau)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(
            "tau = {:.0e}".format(tau)
            + ", score = {:.4f}".format(calculate_score(segment(img), segment(ground)))
        )
        fig.tight_layout()
        fig.savefig("./homework/hw07/2_tau_{:.0e}.png".format(tau))
    return

def test_ADMM_TV():
    img = smooth(512)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./homework/hw07/1_original.png")

def test_all():
    test_ADMM_LASSO()
    test_ADMM_TV()
    return


if __name__ == "__main__":
    test_ADMM_TV()
