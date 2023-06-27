from aomip import ProximalOperators, PGM, POGM, OGM1
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test_POGM():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full", 90
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    nmax = 100

    solvers = {}
    solvers["OGM"] = OGM1(A, sino, x0, nmax=nmax, verbose=True, nonneg=True)
    solvers["PGM_constant"] = PGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().constant,
        BB1=True,
        nonneg=True,
    )
    solvers["PGM_huber"] = PGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().huber,
        BB1=True,
        nonneg=True,
    )
    solvers["PGM_l2"] = PGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().l2,
        BB1=True,
        nonneg=True,
    )
    solvers["POGM_constant"] = POGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().constant,
        BB1=True,
        nonneg=True,
    )
    solvers["POGM_huber"] = POGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().huber,
        BB1=True,
        nonneg=True,
    )
    solvers["POGM_l2"] = POGM(
        A,
        sino,
        x0,
        nmax=nmax,
        verbose=True,
        function=ProximalOperators().l2,
        BB1=True,
        nonneg=True,
    )

    scores = {}
    fig2, ax2 = plt.subplots()
    for solver in solvers.items():
        img, x_vec, _ = solver[1].leastSquares()
        fig1, ax1 = plt.subplots()
        ax1.imshow(img, cmap="gray")
        score = calculate_score(segment(img), segment(ground))
        fig1.suptitle(solver[0] + ", score: {}".format(score))
        fig1.savefig("./homework/hw06/1_recon_{}.png".format(solver[0]))
        ax2.plot([calculate_score(segment(i), segment(ground)) for i in x_vec])
        scores[solver[0]] = "{:.4f}".format(score)
    fig2.suptitle("Scores")
    fig2.legend(solvers.keys(), loc="lower right")
    fig2.savefig("./homework/hw06/1_scores.png")
    print(scores)
    return


def test_all():
    test_POGM()
    return


if __name__ == "__main__":
    test_POGM()
