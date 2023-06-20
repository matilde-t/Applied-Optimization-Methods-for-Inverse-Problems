from aomip import ProximalOperators, PGM, GD, ISTA
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test_operators():
    x = np.linspace(-10, 10)

    # Constant function
    y = ProximalOperators().constant(x)
    plt.figure()
    plt.plot(x, y)
    plt.title("Constant function")
    plt.savefig("./homework/hw05/constant.png")

    # Translation
    y = ProximalOperators(prox_g=ProximalOperators(sigma=4).l2, y=4).translation(x)
    plt.figure()
    plt.plot(x, y)
    plt.title("L2 norm translation, sigma = 4, y = 4")
    plt.savefig("./homework/hw05/translation.png")

    sigma = [0, 2, 4, 6]
    # l2 norm
    fig = plt.figure()
    legend = []
    for s in sigma:
        y = ProximalOperators(sigma=s).l2(x)
        plt.plot(x, y)
        legend.append("sigma = {}".format(s))
    plt.legend(legend)
    fig.suptitle("L2 norm")
    plt.savefig("./homework/hw05/l2.png")

    # Huber
    fig = plt.figure()
    legend = []
    sigma = [2, 4]
    delta = [0.1, 0.5, 1, 5]
    for d in delta:
        for s in sigma:
            y = ProximalOperators(delta=d, sigma=s).huber(x)
            plt.plot(x, y)
            legend.append("d = {}, s = {}".format(d, s))
    plt.legend(legend)
    fig.suptitle("Huber")
    plt.savefig("./homework/hw05/huber.png")

    return


def test_PGM():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full"
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    nmax = 100

    solvers = {}

    solvers["Constant"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators().constant,
        nmax=nmax,
        verbose=True,
        backtrack=True,
    )
    solvers["L2"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(sigma=128).l2,
        nmax=nmax,
        verbose=True,
        backtrack=True,
    )
    solvers["Huber"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(delta=0.1, sigma=128).huber,
        nmax=nmax,
        verbose=True,
        backtrack=True,
    )

    for name, solver in solvers.items():
        fig, ax = plt.subplots(3, 1)
        img, x_vec, l_vec = solver.leastSquares()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Reconstruction")
        ax[1].plot(l_vec)
        ax[1].set_title("lambda")
        ax[2].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
        ax[2].set_title("Error")
        fig.suptitle(
            name
            + ", backtracking, score = {}".format(
                calculate_score(segment(img), segment(ground))
            )
        )
        plt.tight_layout()
        plt.savefig("./homework/hw05/proximal-{}-backtracking.png".format(name))

    solvers["Constant"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators().constant,
        nmax=nmax,
        verbose=True,
        BB2=True,
    )
    solvers["L2"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(sigma=128).l2,
        nmax=nmax,
        verbose=True,
        BB2=True,
    )
    solvers["Huber"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(delta=0.1, sigma=128).huber,
        nmax=nmax,
        verbose=True,
        BB2=True,
    )

    for name, solver in solvers.items():
        fig, ax = plt.subplots(3, 1)
        img, x_vec, l_vec = solver.leastSquares()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Reconstruction")
        ax[1].plot(l_vec)
        ax[1].set_title("lambda")
        ax[2].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
        ax[2].set_title("Error")
        fig.suptitle(
            name
            + ", BB2, score = {}".format(calculate_score(segment(img), segment(ground)))
        )
        plt.tight_layout()
        plt.savefig("./homework/hw05/proximal-{}-BB2.png".format(name))

    return


def test_PGM_fast():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full"
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    nmax = 100

    solvers = {}

    solvers["Constant"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators().constant,
        nmax=nmax,
        verbose=True,
        fast1=True,
    )
    solvers["L2"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(sigma=128).l2,
        nmax=nmax,
        verbose=True,
        fast1=True,
    )
    solvers["Huber"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(delta=0.1, sigma=128).huber,
        nmax=nmax,
        verbose=True,
        fast1=True,
    )

    for name, solver in solvers.items():
        fig, ax = plt.subplots(3, 1)
        img, x_vec, l_vec = solver.leastSquares()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Reconstruction")
        ax[1].plot(l_vec)
        ax[1].set_title("lambda")
        ax[2].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
        ax[2].set_title("Error")
        fig.suptitle(
            name
            + ", fast 1, score = {}".format(
                calculate_score(segment(img), segment(ground))
            )
        )
        plt.tight_layout()
        plt.savefig("./homework/hw05/proximal-{}-fast1.png".format(name))

    solvers["Constant"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators().constant,
        nmax=nmax,
        verbose=True,
        fast2=True,
    )
    solvers["L2"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(sigma=128).l2,
        nmax=nmax,
        verbose=True,
        fast2=True,
    )
    solvers["Huber"] = PGM(
        A,
        sino,
        x0,
        ProximalOperators(delta=0.1, sigma=128).huber,
        nmax=nmax,
        verbose=True,
        fast2=True,
    )

    for name, solver in solvers.items():
        fig, ax = plt.subplots(3, 1)
        img, x_vec, l_vec = solver.leastSquares()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Reconstruction")
        ax[1].plot(l_vec)
        ax[1].set_title("lambda")
        ax[2].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
        ax[2].set_title("Error")
        fig.suptitle(
            name
            + ", fast 2, score = {}".format(
                calculate_score(segment(img), segment(ground))
            )
        )
        plt.tight_layout()
        plt.savefig("./homework/hw05/proximal-{}-fast2.png".format(name))

    return


def test_convergence():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07a_full"
    )
    x_shape = np.array([512, 512])

    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07a_recon.tif"
    )

    nmax = 200

    gd = GD(A, sino, x0, nmax=nmax, verbose=True)
    pgm = PGM(A, sino, x0, ProximalOperators().l2, nmax=nmax, verbose=True)

    fig, ax = plt.subplots(2, 1)
    img, x_vec, _ = gd.l2Norm()
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Reconstruction")
    ax[1].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
    ax[1].set_title("Error")
    fig.suptitle(
        "GD with Tikhonov regularization, score = {}".format(
            calculate_score(segment(img), segment(ground))
        )
    )
    plt.tight_layout()
    plt.savefig("./homework/hw05/conv-GD.png")

    fig, ax = plt.subplots(2, 1)
    img, x_vec, _ = pgm.leastSquares()
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Reconstruction")
    ax[1].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
    ax[1].set_title("Error")
    fig.suptitle(
        "PGM with L2 operator, score = {}".format(
            calculate_score(segment(img), segment(ground))
        )
    )
    plt.tight_layout()
    plt.savefig("./homework/hw05/conv-PGM.png")

    return


def test_elastic_net():
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
    solvers["None"] = ISTA(A, sino, x0, nmax=nmax, verbose=True)
    solvers["Backtracking"] = ISTA(A, sino, x0, backtrack=True, nmax=nmax, verbose=True)
    solvers["BB1"] = ISTA(A, sino, x0, BB1=True, nmax=nmax, verbose=True)
    solvers["BB2"] = ISTA(A, sino, x0, BB2=True, nmax=nmax, verbose=True)

    for name, solver in solvers.items():
        fig, ax = plt.subplots(3, 1)
        img, x_vec, l_vec = solver.leastSquares()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Reconstruction")
        ax[1].plot(l_vec)
        ax[1].set_title("lambda")
        ax[2].plot([np.linalg.norm(x - ground) ** 2 for x in x_vec])
        ax[2].set_title("Error")
        fig.suptitle(
            name + ", score = {}".format(calculate_score(segment(img), segment(ground)))
        )
        plt.tight_layout()
        plt.savefig("./homework/hw05/elastic-net-{}.png".format(name))

    return


def test_all():
    test_operators()
    test_PGM()
    test_PGM_fast()
    test_convergence()
    test_elastic_net()
    return


if __name__ == "__main__":
    test_elastic_net()
