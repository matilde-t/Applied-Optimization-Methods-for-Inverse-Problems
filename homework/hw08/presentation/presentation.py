from aomip import OGM1, ISTA, PGD, GD, load_lowdose_data
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test():
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)
    nmax = 10

    # Helsinki challenge
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full", 30
    )

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    solvers = {}
    solvers["GD"] = GD(A, sino, x0, nmax=nmax)
    solvers["GD_backtrack"] = GD(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["GD_BB1"] = GD(A, sino, x0, nmax=nmax, BB1=True)
    solvers["GD_BB2"] = GD(A, sino, x0, nmax=nmax, BB2=True)
    solvers["GD_circle"] = GD(A, sino, x0, nmax=nmax, circle=True)
    solvers["GD_backtrack_circle"] = GD(A, sino, x0, nmax=nmax, backtrack=True, circle=True)
    solvers["GD_BB1_circle"] = GD(A, sino, x0, nmax=nmax, BB1=True, circle=True)
    solvers["GD_BB2_cricle"] = GD(A, sino, x0, nmax=nmax, BB2=True, circle=True)
    solvers["ISTA"] = ISTA(A, sino, x0, nmax=nmax)
    solvers["ISTA_backtrack"] = ISTA(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["ISTA_BB1"] = ISTA(A, sino, x0, nmax=nmax, BB1=True)
    solvers["ISTA_BB2"] = ISTA(A, sino, x0, nmax=nmax, BB2=True)
    solvers["OGM"] = OGM1(A, sino, x0, nmax=nmax)
    solvers["OGM_nonnegative"] = OGM1(A, sino, x0, nmax=nmax, nonneg=True)
    solvers["PGD"] = PGD(A, sino, x0, nmax=nmax)
    solvers["PGD_backtrack"] = PGD(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["PGD_BB1"] = PGD(A, sino, x0, nmax=nmax, BB1=True)
    solvers["PGD_BB2"] = PGD(A, sino, x0, nmax=nmax, BB2=True)
    solvers["PGD_BB2_circle"] = PGD(A, sino, x0, nmax=nmax, BB2=True, circle=True)

    scores = {}

    for key, solver in solvers.items():
        img = solver.leastSquares()
        scores[key] = "{:.4f}".format(calculate_score(segment(img), segment(ground)))
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.set_title(key + ", score = " + scores[key])
        fig.savefig("./homework/hw08/presentation/helsinki_{}.png".format(key))
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(ground, cmap="gray")
    ax.set_title("Ground truth")
    fig.savefig("./homework/hw08/presentation/helsinki_ground.png")

    print(scores)

    # Low dose dataset
    _, _, ground = load_lowdose_data(
        "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_fd.tif",
        10,
    )
    sino, A, img = load_lowdose_data(
        "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_qd.tif",
        10,
    )

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(
        "Filtered backprojection, score = {:.4f}".format(
            calculate_score(segment(img), segment(ground))
        )
    )
    fig.savefig("./homework/hw08/presentation/low_FBP.png")

    solvers = {}
    solvers["GD"] = GD(A, sino, x0, nmax=nmax)
    solvers["GD_backtrack"] = GD(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["GD_BB1"] = GD(A, sino, x0, nmax=nmax, BB1=True)
    solvers["GD_BB2"] = GD(A, sino, x0, nmax=nmax, BB2=True)
    solvers["ISTA"] = ISTA(A, sino, x0, nmax=nmax)
    solvers["ISTA_backtrack"] = ISTA(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["ISTA_BB1"] = ISTA(A, sino, x0, nmax=nmax, BB1=True)
    solvers["ISTA_BB2"] = ISTA(A, sino, x0, nmax=nmax, BB2=True)
    solvers["OGM"] = OGM1(A, sino, x0, nmax=nmax)
    solvers["OGM_nonnegative"] = OGM1(A, sino, x0, nmax=nmax, nonneg=True)
    solvers["PGD"] = PGD(A, sino, x0, nmax=nmax)
    solvers["PGD_backtrack"] = PGD(A, sino, x0, nmax=nmax, backtrack=True)
    solvers["PGD_BB1"] = PGD(A, sino, x0, nmax=nmax, BB1=True)
    solvers["PGD_BB2"] = PGD(A, sino, x0, nmax=nmax, BB2=True)

    scores = {}

    for key, solver in solvers.items():
        img = solver.leastSquares()
        scores[key] = "{:.4f}".format(calculate_score(segment(img), segment(ground)))
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.set_title(key + ", score = " + scores[key])
        fig.savefig("./homework/hw08/presentation/low_{}.png".format(key))
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(ground, cmap="gray")
    ax.set_title("Ground truth")
    fig.savefig("./homework/hw08/presentation/low_ground.png")

    print(scores)
    return


if __name__ == "__main__":
    test()
