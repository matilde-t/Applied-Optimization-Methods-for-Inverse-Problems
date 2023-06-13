from aomip import GD, ISTA, PGD, shepp_logan, radon, noise, XrayOperator
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test_linesearch():
    gd = {}
    gd["Line Search"] = GD(backtrack=True)
    gd["Barzilai and Borwein 1"] = GD(BB1=True)
    gd["Barzilai and Borwein 2"] = GD(BB2=True)

    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    x0 = np.array([-5, -5])

    ## Booth function
    booth = lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    dbooth = lambda x: np.array([10 * x[0] + 8 * x[1] - 34, 8 * x[0] + 10 * x[1] - 38])
    Z = booth([X, Y])
    plt.figure()
    plt.contour(X, Y, Z, 1000)
    plt.colorbar()
    plt.scatter(1, 3, c="r")
    plt.title("Booth function")
    plt.savefig("./homework/hw04/booth.png")
    print("Booth function")

    booth_err = {}
    booth_res = {}
    for pair in gd.items():
        booth_res[pair[0]] = pair[1].gradDesc(dbooth, booth, x0)
        booth_err[pair[0]] = np.linalg.norm(booth_res[pair[0]] - np.array([1, 3]))

    print("Booth function results")
    print(booth_res)
    print("Booth function errors")
    print(booth_err)

    ## Three-hump camel function
    camel = (
        lambda x: 2 * x[0] ** 2
        - 1.05 * x[0] ** 4
        + x[0] ** 6 / 6
        + x[0] * x[1]
        + x[1] ** 2
    )
    dcamel = lambda x: np.array(
        [4 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1], x[0] + 2 * x[1]]
    )
    Z = camel([X, Y])
    plt.figure()
    plt.contour(X, Y, Z, 1000)
    plt.colorbar()
    plt.scatter(0, 0, c="r")
    plt.title("Three-hump camel function")
    plt.savefig("./homework/hw04/camel.png")
    print("Camel function")

    camel_err = {}
    camel_res = {}
    for pair in gd.items():
        camel_res[pair[0]] = pair[1].gradDesc(dcamel, camel, x0)
        camel_err[pair[0]] = np.linalg.norm(camel_res[pair[0]])

    print("Camel function results")
    print(camel_res)
    print("Camel function errors")
    print(camel_err)

    ## Helsinki
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full"
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(sino, cmap="gray")
    ax[0].set_title("Sinogram")
    ax[1].imshow(ground, cmap="gray")
    ax[1].set_title("Ground truth")
    fig.suptitle("Helsinki original data")
    plt.tight_layout()
    plt.savefig("./homework/hw04/htc2022_orig.png")

    nmax = 300
    gd = {}
    gd["Line_Search"] = GD(A, sino, x0, verbose=True, backtrack=True, nmax=nmax)
    gd["Barzilai_and_Borwein_1"] = GD(A, sino, x0, verbose=True, BB1=True, nmax=nmax)
    gd["Barzilai_and_Borwein_2"] = GD(A, sino, x0, verbose=True, BB2=True, nmax=nmax)

    res_helsinki = {}
    score_helsinki = {}
    x_helsinki = {}
    l_helsinki = {}
    for pair in gd.items():
        res_helsinki[pair[0]], x_helsinki[pair[0]], l_helsinki[pair[0]] = pair[
            1
        ].leastSquares()
        score_helsinki[pair[0]] = calculate_score(
            segment(res_helsinki[pair[0]]), segment(ground)
        )

    print("Helsinki scores")
    print(score_helsinki)

    for pair in gd.items():
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(res_helsinki[pair[0]], cmap="gray")
        ax[0, 0].set_title(
            "Final iteration, score: {:.2f}".format(score_helsinki[pair[0]])
        )
        ax[0, 1].plot(l_helsinki[pair[0]])
        ax[0, 1].set_xlabel("Iteration")
        ax[0, 1].set_title("Values of lambda")
        ax[1, 0].imshow(x_helsinki[pair[0]][10], cmap="gray")
        ax[1, 0].set_title(
            "100th iteration, score: {:.2f}".format(
                calculate_score(segment(x_helsinki[pair[0]][100]), segment(ground))
            )
        )
        ax[1, 1].imshow(x_helsinki[pair[0]][50], cmap="gray")
        ax[1, 1].set_title(
            "200th iteration, score: {:.2f}".format(
                calculate_score(segment(x_helsinki[pair[0]][200]), segment(ground))
            )
        )
        fig.suptitle(pair[0])
        plt.tight_layout()
        plt.savefig("./homework/hw04/htc2022_{}.png".format(pair[0]))

    fig, ax = plt.subplots(3, 1)
    i = 0
    for pair in gd.items():
        ax[i].plot([np.linalg.norm(el - ground) ** 2 for el in x_helsinki[pair[0]]])
        ax[i].set_title(pair[0])
        ax[i].set_ylim([1.81e8, 1.826e8])
        ax[i].set_xlabel("Iteration")
        i += 1
    fig.suptitle("Convergence analysis (squared 2-norm error)")
    plt.tight_layout()
    plt.savefig("./homework/hw04/htc2022_convergence.png")

    return


def test_ISTA():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full"
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    nmax = 300
    ista = {}
    ista["Default"] = ISTA(A, sino, x0, verbose=True, nmax=nmax)
    ista["Line_Search"] = ISTA(A, sino, x0, verbose=True, backtrack=True, nmax=nmax)
    ista["Barzilai_and_Borwein_1"] = ISTA(
        A, sino, x0, verbose=True, BB1=True, nmax=nmax
    )
    ista["Barzilai_and_Borwein_2"] = ISTA(
        A, sino, x0, verbose=True, BB2=True, nmax=nmax
    )

    res_helsinki = {}
    score_helsinki = {}
    x_helsinki = {}
    l_helsinki = {}
    for pair in ista.items():
        res_helsinki[pair[0]], x_helsinki[pair[0]], l_helsinki[pair[0]] = pair[
            1
        ].leastSquares()
        score_helsinki[pair[0]] = calculate_score(
            segment(res_helsinki[pair[0]]), segment(ground)
        )

    print("Helsinki scores")
    print(score_helsinki)

    for pair in ista.items():
        ## Images
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(res_helsinki[pair[0]], cmap="gray")
        ax[0, 0].set_title(
            "Final iteration, score: {:.2f}".format(score_helsinki[pair[0]])
        )
        ax[0, 1].plot(l_helsinki[pair[0]])
        ax[0, 1].set_xlabel("Iteration")
        ax[0, 1].set_title("Values of lambda")
        ax[1, 0].imshow(x_helsinki[pair[0]][10], cmap="gray")
        ax[1, 0].set_title(
            "100th iteration, score: {:.2f}".format(
                calculate_score(segment(x_helsinki[pair[0]][100]), segment(ground))
            )
        )
        ax[1, 1].imshow(x_helsinki[pair[0]][50], cmap="gray")
        ax[1, 1].set_title(
            "200th iteration, score: {:.2f}".format(
                calculate_score(segment(x_helsinki[pair[0]][200]), segment(ground))
            )
        )
        fig.suptitle(pair[0])
        plt.tight_layout()
        plt.savefig("./homework/hw04/ISTA_{}.png".format(pair[0]))

    ## Convergence
    fig = plt.figure()
    legend = [
        "Default",
        "Line Search",
        "Barzilai and Borwein 1",
        "Barzilai and Borwein 2",
    ]
    for key in ista.keys():
        plt.plot([np.linalg.norm(el - ground) ** 2 for el in x_helsinki[key]])
    plt.xlabel("Iteration")
    plt.legend(legend)
    fig.suptitle("Convergence analysis (squared 2-norm error)")
    plt.tight_layout()
    plt.savefig("./homework/hw04/ISTA_convergence.png")

    return


def test_PGD():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full", 90
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    nmax = 100
    lb = [-np.inf, 0]
    ub = [50, 150, 250, np.inf]
    for low in lb:
        for up in ub:
            c = [low, up]
            pgd = {}
            pgd["Default"] = PGD(A, sino, x0, verbose=True, nmax=nmax, c=c)
            pgd["Line_Search"] = PGD(
                A, sino, x0, verbose=True, backtrack=True, nmax=nmax, c=c
            )
            pgd["Barzilai_and_Borwein_1"] = PGD(
                A, sino, x0, verbose=True, BB1=True, nmax=nmax, c=c
            )
            pgd["Barzilai_and_Borwein_2"] = PGD(
                A, sino, x0, verbose=True, BB2=True, nmax=nmax, c=c
            )

            res_helsinki = {}
            score_helsinki = {}
            x_helsinki = {}

            ## Convergence
            fig = plt.figure()
            legend = []
            for pair in pgd.items():
                res_helsinki[pair[0]], x_helsinki[pair[0]], _ = pair[1].leastSquares()
                score_helsinki[pair[0]] = calculate_score(
                    segment(res_helsinki[pair[0]]), segment(ground)
                )
                plt.plot(
                    [np.linalg.norm(el - ground) ** 2 for el in x_helsinki[pair[0]]]
                )
                legend.append(
                    pair[0] + ", final score: {:.2f}".format(score_helsinki[pair[0]])
                )
            plt.xlabel("Iteration")
            fig.suptitle(
                "Convergence analysis (squared 2-norm error), c = {}".format(c)
            )
            plt.legend(legend)
            plt.tight_layout()
            plt.savefig("./homework/hw04/PDG_convergence_{}-{}.png".format(c[0], c[1]))

            ## Images
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(res_helsinki["Default"], cmap="gray")
            ax[0, 0].set_title(
                "Default, score: {:.2f}".format(score_helsinki["Default"])
            )
            ax[0, 1].imshow(res_helsinki["Line_Search"], cmap="gray")
            ax[0, 1].set_title(
                "Line search, score: {:.2f}".format(score_helsinki["Line_Search"])
            )
            ax[1, 0].imshow(res_helsinki["Barzilai_and_Borwein_1"], cmap="gray")
            ax[1, 0].set_title(
                "BB1, score: {:.2f}".format(score_helsinki["Barzilai_and_Borwein_1"])
            )
            ax[1, 1].imshow(res_helsinki["Barzilai_and_Borwein_2"], cmap="gray")
            ax[1, 1].set_title(
                "BB2, score: {:.2f}".format(score_helsinki["Barzilai_and_Borwein_2"])
            )
            fig.suptitle("Final results, c = {}".format(c))
            plt.tight_layout()
            plt.savefig("./homework/hw04/PDG_final_{}-{}.png".format(c[0], c[1]))

            print("Helsinki scores, c = {}".format(c))
            print(score_helsinki)

    tifffile.imwrite(
        "./homework/hw04/htc2022_07c_90_BB2.tif", res_helsinki["Barzilai_and_Borwein_2"]
    )

    return


def test_dataset():
    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    sino = {}
    A = {}
    solutions = {"GD": {}, "ISTA": {}, "PGD": {}}
    scores = {"GD": {}, "ISTA": {}, "PGD": {}}
    angles = [90, 60, 30]
    solvers = ["GD", "ISTA", "PGD"]
    nmax = 1000
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)

    for angle in angles:
        sino[angle], A[angle] = load_htc2022data(
            "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full", angle
        )
        solutions["GD"][angle] = GD(
            A[angle], sino[angle], x0, nmax=nmax, BB2=True
        ).leastSquares()
        solutions["ISTA"][angle] = ISTA(
            A[angle], sino[angle], x0, nmax=nmax, BB2=True
        ).leastSquares()
        solutions["PGD"][angle] = PGD(
            A[angle], sino[angle], x0, nmax=nmax, BB2=True
        ).leastSquares()

    for solver in solvers:
        fig, ax = plt.subplots(1, 3)
        i = 0
        for angle in angles:
            scores[solver][angle] = calculate_score(
                segment(solutions[solver][angle]), segment(ground)
            )
            ax[i].imshow(solutions[solver][angle], cmap="gray")
            ax[i].set_title(
                "{}° arc, score: {:.2f}".format(angle, scores[solver][angle])
            )
            i += 1
            tifffile.imwrite(
                "./homework/hw04/htc2022_07a_{}_{}.tif".format(solver, angle),
                solutions[solver][angle],
            )
        fig.suptitle("Final results, {}".format(solver))
        plt.tight_layout()
        plt.savefig("./homework/hw04/htc2022_07a_{}.png".format(solver))

    print("Scores")
    print(scores)

    return


def test_semiConv():
    arc = 360
    angles = 420
    sino_shape = [420]
    shape = [512, 512]
    phantom = shepp_logan(shape)
    sino = radon(phantom, sino_shape, np.linspace(0, arc, angles), 1000, 150)
    noisy_sino = noise(sino).gaussian()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(phantom, cmap="gray")
    ax[0].set_title("Phantom")
    ax[1].imshow(sino, cmap="gray")
    ax[1].set_title("Sinogram")
    ax[2].imshow(noisy_sino, cmap="gray")
    ax[2].set_title("Gaussian noise")
    fig.suptitle("Forward projection")
    plt.tight_layout()
    plt.savefig("./homework/hw04/forwardProjection.png")
    A = XrayOperator(shape, sino_shape, np.linspace(0, arc, angles), 1000, 150)
    x0 = np.zeros(shape)
    nmax = 30

    solvers = {}
    solvers["GD, default"] = GD(A, noisy_sino, x0, nmax=nmax, verbose=True)
    solvers["ISTA, default"] = ISTA(A, noisy_sino, x0, nmax=nmax, verbose=True)
    solvers["PGD, default"] = PGD(A, noisy_sino, x0, nmax=nmax, verbose=True)
    solvers["GD, backtracking"] = GD(
        A, noisy_sino, x0, nmax=nmax, verbose=True, backtrack=True
    )
    solvers["ISTA, backtracking"] = ISTA(
        A, noisy_sino, x0, nmax=nmax, verbose=True, backtrack=True
    )
    solvers["PGD, backtracking"] = PGD(
        A, noisy_sino, x0, nmax=nmax, verbose=True, backtrack=True
    )
    solvers["GD, BB1"] = GD(A, noisy_sino, x0, nmax=nmax, verbose=True, BB1=True)
    solvers["ISTA, BB1"] = ISTA(A, noisy_sino, x0, nmax=nmax, verbose=True, BB1=True)
    solvers["PGD, BB1"] = PGD(A, noisy_sino, x0, nmax=nmax, verbose=True, BB1=True)

    solutions = {}
    evolutions = {}
    for solver in solvers.items():
        solutions[solver[0]], evolutions[solver[0]], _ = solver[1].leastSquares()
    fig = plt.figure()
    legend = []
    for solver in solvers.keys():
        plt.plot([np.linalg.norm(el - phantom) ** 2 for el in evolutions[solver]])
        legend.append(solver)
    plt.xlabel("Iteration")
    fig.suptitle("Convergence analysis (squared 2-norm error)")
    plt.tight_layout(rect=[0, 0, 0.6, 1])
    plt.legend(legend, bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig("./homework/hw04/semi_convergence.png")
    return


def test_all():
    test_linesearch()
    test_ISTA()
    test_PGD()
    test_dataset()
    test_semiConv()
    return


if __name__ == "__main__":
    test_all()
