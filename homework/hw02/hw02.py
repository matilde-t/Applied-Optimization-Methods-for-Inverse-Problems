import aomip
import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image


def test_Slicing():
    path = "/srv/ceph/share-all/aomip/6983008_seashell"
    img = []
    for image in os.listdir(path):
        tmp = aomip.centerOfRotationCorrection(
            tifffile.imread(os.path.join(path, image)), -4, 1
        )  # correction as suggested in the dataset
        img.append(tmp)
    sino = []
    idx = []
    for i in range(0, 2240, 500):
        sino.append(aomip.slice(img, i))
        idx.append(i)
    n = len(sino)
    fig, ax = plt.subplots(1, n - 1)
    for i in range(1, n):
        ax[i - 1].imshow(sino[i], cmap="gray")
        ax[i - 1].set_title("Row {}".format(idx[i]))
    fig.suptitle("Sinograms of different rows")
    plt.tight_layout()
    plt.savefig("./homework/hw02/sinogram.png")
    return


def printShell():
    img = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    img = aomip.centerOfRotationCorrection(img, -4, 1)
    img = aomip.binning(img, 3)
    plt.imsave("./homework/hw02/seashell.png", img, cmap="gray")
    return


def test_GradientDescent():
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    x0 = np.array([-5, -5])
    l_val = [1, 1e-1, 1e-2, 1e-3, 1e-4]

    booth = lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    dbooth = lambda x: np.array([10 * x[0] + 8 * x[1] - 34, 8 * x[0] + 10 * x[1] - 38])
    Z = booth([X, Y])
    plt.figure()
    plt.contour(X, Y, Z, 1000)
    plt.colorbar()
    plt.scatter(1, 3, c="r")
    plt.title("Booth function")
    plt.savefig("./homework/hw02/booth.png")
    print("Booth function")
    res = {}
    for l in l_val:
        x = aomip.gradDesc(dbooth, x0, l)
        res[l] = x
    print(res)

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
    plt.savefig("./homework/hw02/camel.png")
    print("Camel function")
    res = {}
    for l in l_val:
        x = aomip.gradDesc(dcamel, x0, l, 1e7)
        res[l] = x
    print(res)
    return


def test_LeastSquares():
    arc = 360
    angles = 70
    thetas = np.linspace(0, arc, angles)
    sino_shape = [420]
    s2c = 1000
    c2d = 150
    phantom = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    phantom = aomip.centerOfRotationCorrection(phantom, -4, 1)
    phantom = aomip.binning(phantom, 3)
    x_shape = phantom.shape
    sino = aomip.radon(phantom, sino_shape, thetas, s2c, c2d)
    A = aomip.XrayOperator(x_shape, sino_shape, thetas, s2c, c2d)

    plt.figure()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom, cmap="gray")
    ax[0].set_title("Original image")
    ax[1].imshow(sino, cmap="gray")
    ax[1].set_title("Sinogram")
    plt.tight_layout()
    fig.suptitle("Original image and sinogram")
    plt.savefig("./homework/hw02/leastSquares_orig.png")

    x0 = np.ones(phantom.shape)
    l_val = [1e-3, 1e-4, 1e-5]
    res = {}
    for l in l_val:
        res[l] = aomip.leastSquares(A, sino, x0, l, 1e4)
    plt.figure()
    fig, ax = plt.subplots(1, len(l_val))
    for i, l in enumerate(res):
        ax[i].imshow(res[l], cmap="gray")
        ax[i].set_title("lambda = {}".format(l))
    fig.suptitle("Reconstructed images")
    plt.tight_layout()
    plt.savefig("./homework/hw02/leastSquares.png")

    err = {}
    for l in res.keys():
        err[l] = np.linalg.norm(res[l] - phantom)
    plt.figure()
    plt.scatter(err.keys(), err.values(), c="b")
    plt.plot(err.keys(), err.values())
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.title("Error as a function of lambda")
    plt.savefig("./homework/hw02/leastSquares_err.png")
    return


def test_l2Norm():
    arc = 360
    angles = 70
    thetas = np.linspace(0, arc, angles)
    sino_shape = [420]
    s2c = 1000
    c2d = 150
    phantom = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    phantom = aomip.centerOfRotationCorrection(phantom, -4, 1)
    phantom = aomip.binning(phantom, 3)
    x_shape = phantom.shape
    sino = aomip.radon(phantom, sino_shape, thetas, s2c, c2d)
    A = aomip.XrayOperator(x_shape, sino_shape, thetas, s2c, c2d)
    x0 = np.ones(phantom.shape)
    beta_list = np.arange(-0.9, 1, 0.2)
    res = {}
    err_phantom = {}
    for beta in beta_list:
        x = aomip.l2Norm(A, sino, x0, beta=beta, nmax=1e4)
        res[beta] = x
        err_phantom[beta] = np.linalg.norm(x - phantom)
    plt.figure()
    fig, ax = plt.subplots(len(beta_list) // 2, 2, figsize=(15, 15))
    for i, beta in enumerate(res):
        ax[i // 2][i - 2 * (i // 2)].imshow(res[beta], cmap="gray")
        ax[i // 2][i - 2 * (i // 2)].set_title("beta = {:.4}".format(beta))
    fig.suptitle("Reconstructed images using Tikhonov regularization")
    plt.tight_layout()
    plt.savefig("./homework/hw02/l2Norm.png")

    plt.figure()
    plt.scatter(err_phantom.keys(), err_phantom.values(), c="b")
    plt.plot(err_phantom.keys(), err_phantom.values())
    plt.title("Error as a function of beta")
    plt.yscale("log")
    plt.xlabel("beta")
    plt.ylabel("error")
    plt.tight_layout()
    plt.savefig("./homework/hw02/l2Norm_err.png")
    return


def test_huber():
    arc = 360
    angles = 70
    thetas = np.linspace(0, arc, angles)
    sino_shape = [420]
    s2c = 1000
    c2d = 150
    phantom = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    phantom = aomip.centerOfRotationCorrection(phantom, -4, 1)
    phantom = aomip.binning(phantom, 3)
    x_shape = phantom.shape
    sino = aomip.radon(phantom, sino_shape, thetas, s2c, c2d)
    A = aomip.XrayOperator(x_shape, sino_shape, thetas, s2c, c2d)
    x0 = np.ones(phantom.shape)
    beta = -0.1
    delta_list = [100, 200]
    res = {}
    err_phantom = {}
    for delta in delta_list:
        x = aomip.huber(A, sino, x0, beta=beta, nmax=1e3, delta=delta)
        res[delta] = x
        err_phantom[delta] = np.linalg.norm(x - phantom)
    # plt.figure()
    # fig, ax = plt.subplots(len(delta_list)//2,2, figsize=(15,15))
    # for i, delta in enumerate(res):
    #     ax[i//2][i-2*(i//2)].imshow(res[delta], cmap='gray')
    #     ax[i//2][i-2*(i//2)].set_title('delta = {:.4}'.format(delta))
    # fig.suptitle('Reconstructed images using Tikhonov regularization')
    # plt.tight_layout()
    # plt.savefig('./homework/hw02/huber.png')

    plt.figure()
    plt.scatter(err_phantom.keys(), err_phantom.values(), c="b")
    plt.plot(err_phantom.keys(), err_phantom.values())
    plt.title("Error as a function of delta")
    plt.yscale("log")
    plt.xlabel("beta")
    plt.ylabel("error")
    plt.tight_layout()
    plt.savefig("./homework/hw02/huber_err.png")

    return


def test_fair():
    arc = 360
    angles = 70
    thetas = np.linspace(0, arc, angles)
    sino_shape = [420]
    s2c = 1000
    c2d = 150
    phantom = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    phantom = aomip.centerOfRotationCorrection(phantom, -4, 1)
    phantom = aomip.binning(phantom, 3)
    x_shape = phantom.shape
    sino = aomip.radon(phantom, sino_shape, thetas, s2c, c2d)
    A = aomip.XrayOperator(x_shape, sino_shape, thetas, s2c, c2d)
    x0 = np.ones(phantom.shape)
    beta = -0.1
    delta_list = [1e12]
    res = {}
    err_phantom = {}
    for delta in delta_list:
        x = aomip.huber(A, sino, x0, beta=beta, nmax=1e3, delta=delta)
        res[delta] = x
        err_phantom[delta] = np.linalg.norm(x - phantom)
    # plt.figure()
    # fig, ax = plt.subplots(len(delta_list)//2,2, figsize=(15,15))
    # for i, delta in enumerate(res):
    #     ax[i//2][i-2*(i//2)].imshow(res[delta], cmap='gray')
    #     ax[i//2][i-2*(i//2)].set_title('delta = {:.4}'.format(delta))
    # fig.suptitle('Reconstructed images using Tikhonov regularization')
    # plt.tight_layout()
    # plt.savefig('./homework/hw02/fair.png')

    plt.figure()
    plt.scatter(err_phantom.keys(), err_phantom.values(), c="b")
    plt.plot(err_phantom.keys(), err_phantom.values())
    plt.title("Error as a function of delta")
    plt.yscale("log")
    plt.xlabel("beta")
    plt.ylabel("error")
    plt.tight_layout()
    plt.savefig("./homework/hw02/fair_err.png")


def test_L():
    arc = 360
    angles = 70
    thetas = np.linspace(0, arc, angles)
    sino_shape = [420]
    s2c = 1000
    c2d = 150
    phantom = tifffile.imread(
        "/srv/ceph/share-all/aomip/6983008_seashell/20211124_seashell_0666.tif"
    )
    phantom = aomip.centerOfRotationCorrection(phantom, -4, 1)
    phantom = aomip.binning(phantom, 3)
    x_shape = phantom.shape
    sino = aomip.radon(phantom, sino_shape, thetas, s2c, c2d)
    A = aomip.XrayOperator(x_shape, sino_shape, thetas, s2c, c2d)
    x0 = np.ones(phantom.shape)
    L = aomip.forwardDiff(x0)
    x = aomip.l2Norm(A, sino, x0, beta=1, nmax=1e4, L=L)
    err_phantom = np.linalg.norm(x - phantom)
    plt.figure()
    plt.imshow(x, cmap="gray")
    plt.title("Reconstructed images using forward difference operator")
    plt.tight_layout()
    plt.savefig("./homework/hw02/diff.png")
    print("Error: {:.5E}".format(err_phantom))
    return


def test_Reconstruction():
    data = loadmat(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_01c_full",
        simplify_cells=True,
        squeeze_me=True,
    )
    sino = data["CtDataFull"]["sinogram"].T
    angles = data["CtDataFull"]["parameters"]["angles"]
    source_origin = data["CtDataFull"]["parameters"]["distanceSourceOrigin"] * 100
    source_detector = data["CtDataFull"]["parameters"]["distanceSourceDetector"]
    x_shape = np.array([512, 512])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(sino, cmap="gray")
    ax[0].set_title("Sinogram")
    ax[1].imshow(
        np.asarray(
            Image.open(
                "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_01c_recon_fbp.png"
            )
        ),
        cmap="gray",
    )
    ax[1].set_title("Reference image")
    fig.suptitle("Helsinki original data")
    plt.tight_layout()
    plt.savefig("./homework/hw02/htc2022_orig.png")

    res = []
    filters = ["none", "ram-lak", "shepp-logan", "cosine"]
    for filter in filters:
        res.append(
            aomip.iradon(
                sino,
                [sino.shape[0]],
                x_shape,
                angles,
                source_origin,
                source_detector,
                filter=filter,
            )
        )

    A = aomip.XrayOperator(
        x_shape, [sino.shape[0]], angles, source_origin, source_detector
    )
    x0 = np.zeros(x_shape)
    res.append(aomip.leastSquares(A, sino, x0))
    res.append(aomip.l2Norm(A, sino, x0))

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(0, 4):
        ax[i // 2][i - 2 * (i // 2)].imshow(res[i], cmap="gray")
        ax[i // 2][i - 2 * (i // 2)].set_title("Filter: {}".format(filters[i]))
    for i in range(4, 6):
        ax[i // 2][i - 2 * (i // 2)].imshow(res[i], cmap="gray")
    ax[2][0].set_title("Least squares")
    ax[2][1].set_title("Tikhonov")
    fig.suptitle("Reconstructed images")
    plt.tight_layout()
    plt.savefig("./homework/hw02/htc2022_recon.png")
    return


def test_all():
    test_Slicing()
    printShell()
    test_GradientDescent()
    test_LeastSquares()
    test_l2Norm()
    test_L()
    test_Reconstruction()
    return


if __name__ == "__main__":
    test_Reconstruction()
