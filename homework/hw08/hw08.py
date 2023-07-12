from aomip import SGM, ADMM, ISTA
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test_SGM():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07b_full", 60
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07b_recon.tif"
    )

    nmax = 500

    solvers = {}
    solvers["SGM_constant"] = SGM(A, sino, x0, nmax=nmax, verbose=True)
    solvers["SGM_decreasing"] = SGM(
        A, sino, x0, nmax=nmax, verbose=True, constant=False
    )

    for name, solver in solvers.items():
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
        i = 0
        for beta in np.logspace(-3, 2, 6):
            img, x, _ = solver.l1Norm(beta)
            ax[i // 3, i % 3].imshow(img, cmap="gray")
            ax[i // 3, i % 3].axis("off")
            ax1[i // 3, i % 3].plot(
                [calculate_score(segment(a), segment(ground)) for a in x]
            )
            ax1[i // 3, i % 3].set_title("beta = {:.0e}".format(beta))
            ax1[i // 3, i % 3].set_ylim([0, 0.5])
            score = calculate_score(segment(img), segment(ground))
            ax[i // 3, i % 3].set_title(
                "beta = {:.0e}".format(beta) + ", score = {:.4f}".format(score)
            )
            i += 1
        fig.suptitle(name + " reconstruction")
        fig.tight_layout()
        fig.savefig("./homework/hw08/1_{}.png".format(name))
        fig1.suptitle(name + " scores")
        fig1.tight_layout()
        fig1.savefig("./homework/hw08/1_{}_err.png".format(name))
    return


def test_ADMM_TV():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07b_full", 60
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07b_recon.tif"
    )

    x0 = np.zeros(x_shape)

    nmax = 100

    for tau in np.logspace(-3, 6, 10):
        img = ADMM(A, sino, x0, nmax=nmax).TV_isotropic(tau=tau)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(
            "tau = {:.0e}".format(tau)
            + ", score = {:.4f}".format(calculate_score(segment(img), segment(ground)))
        )
        fig.tight_layout()
        fig.savefig("./homework/hw08/2_tau_{:.0e}_iso.png".format(tau))
    return


def test_challenge(angle):
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full", angle
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    x0 = np.zeros(x_shape)

    nmax = 400

    _, x, _ = ISTA(
        A, sino, x0, nmax=nmax, BB2=True, circle=True, nonneg=True, verbose=True
    ).leastSquares()
    scores = [calculate_score(segment(a), segment(ground)) for a in x]
    idx = scores.index(max(scores))
    tifffile.imwrite("./homework/hw08/3_07c_recon_{}.tif".format(angle), x[idx])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(x[idx], cmap="gray")
    ax.axis("off")
    ax.set_title(
        "angle = {}".format(angle)
        + ", score = {:.4f}".format(max(scores))
        + ", iteration = {}".format(idx)
    )
    fig.tight_layout()
    fig.savefig("./homework/hw08/3_07c_recon_{}.png".format(angle))
    print(angle, max(scores), idx)
    return


if __name__ == "__main__":
    for angle in [360, 90, 60, 30]:
        test_challenge(angle)
