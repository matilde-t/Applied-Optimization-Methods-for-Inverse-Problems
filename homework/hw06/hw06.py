from aomip import (
    ProximalOperators,
    PGM,
    POGM,
    OGM1,
    load_lowdose_data,
    absorptionToTransmission,
    GD,
    ISTA,
)
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


def test_lowDose():
    errors = []
    for i in [0, 10, 20, 30]:
        fsino, _, fimg = load_lowdose_data(
            "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_fd.tif",
            i,
        )
        ffig, fax = plt.subplots(2, 1)
        fax[0].imshow(fsino, cmap="gray")
        fax[0].set_title("{}-th slice".format(i))
        fax[1].imshow(fimg, cmap="gray")
        ffig.suptitle("Full dose")
        ffig.tight_layout()
        ffig.savefig("./homework/hw06/2_full_dose_{}.png".format(i))

        qsino, A_, qimg = load_lowdose_data(
            "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_qd.tif",
            i,
        )
        qfig, qax = plt.subplots(2, 1)
        qax[0].imshow(qsino, cmap="gray")
        qax[0].set_title("{}-th slice".format(i))
        qax[1].imshow(qimg, cmap="gray")
        qfig.suptitle("Low dose")
        qfig.tight_layout()
        qfig.savefig("./homework/hw06/2_low_dose_{}.png".format(i))

        errors.append(np.linalg.norm(fimg - qimg) / np.linalg.norm(fimg) * 100)
    print(errors)

    _, _, ground = load_lowdose_data(
        "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_fd.tif",
        10,
    )
    qsino, A, qimg = load_lowdose_data(
        "/srv/ceph/share-all/aomip/mayo_clinical/out/L067_flat_fan_projections_qd.tif",
        10,
    )
    x0 = np.zeros(qimg.shape)

    scores = {}
    scores["Backprojection"] = calculate_score(segment(ground), segment(qimg))
    solvers = {}
    solvers["GD_backtrack"] = GD(A, qsino, x0, nmax=100, backtrack=True)
    solvers["GD_BB1"] = GD(A, qsino, x0, nmax=100, BB1=True)
    solvers["ISTA_backtrack"] = ISTA(A, qsino, x0, nmax=100, backtrack=True)
    solvers["ISTA_BB1"] = ISTA(A, qsino, x0, nmax=100, BB1=True)

    fig, ax = plt.subplots(2, 2)
    i = 0
    for solver in solvers.items():
        img = solver[1].leastSquares()
        scores[solver[0]] = calculate_score(segment(ground), segment(img))
        ax[i // 2, i % 2].imshow(img, cmap="gray")
        ax[i // 2, i % 2].set_title(
            solver[0] + ", score: {:.4f}".format(scores[solver[0]])
        )
        i += 1
    fig.suptitle("Reconstructions, least squares")
    fig.tight_layout()
    fig.savefig("./homework/hw06/2_scores_LS.png")
    print(scores)

    qsino = normalize(qsino)
    qsino = absorptionToTransmission(qsino)
    df = lambda x: A.applyAdjoint(qsino - np.exp(-A.apply(x))).flatten()
    scores = {}
    scores["Backprojection"] = calculate_score(segment(ground), segment(qimg))
    solvers = {}
    solvers["GD"] = GD(x0=x0, nmax=100)
    solvers["ISTA"] = ISTA(x0=x0, nmax=100)

    fig, ax = plt.subplots(2, 1)
    i = 0
    for solver in solvers.items():
        if solver[0][0:2] == "GD":
            img = solver[1].gradDesc(df)
        else:
            img = solver[1].ISTA(df)
        scores[solver[0]] = calculate_score(segment(ground), segment(img))
        ax[i].imshow(img, cmap="gray")
        ax[i].set_title(solver[0] + ", score: {:.4f}".format(scores[solver[0]]))
        i += 1
    fig.suptitle("Reconstructions, log likelihood")
    fig.tight_layout()
    fig.savefig("./homework/hw06/2_scores_LL.png")
    print(scores)
    return


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def test_all():
    test_POGM()
    test_lowDose()
    return


if __name__ == "__main__":
    test_all()
