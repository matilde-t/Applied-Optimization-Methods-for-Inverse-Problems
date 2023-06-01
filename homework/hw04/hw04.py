from aomip import GD
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

    gd = {}
    gd["Line Search"] = GD(A, sino, x0, backtrack=True, debug=True)
    gd["Barzilai and Borwein 1"] = GD(A, sino, x0, debug=True, BB1=True)
    gd["Barzilai and Borwein 2"] = GD(A, sino, x0, debug=True, BB2=True)

    res_helsinki = {}
    err_helsinki = {}
    x_helsinki = {}
    l_helsinki = {}
    for pair in gd.items():
        res_helsinki[pair[0]], x_helsinki[pair[0]], l_helsinki[pair[0]] = pair[
            1
        ].leastSquares()
        err_helsinki[pair[0]] = np.linalg.norm(res_helsinki[pair[0]] - ground)

    return


def test_all():
    test_linesearch()
    return


if __name__ == "__main__":
    test_linesearch()
