import os
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from aomip import OGM1, GD, LW, CGD, noise, gradDesc, GFconvolution, GFdeconvolution
from challenge.utils import load_htc2022data, segment, calculate_score
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def test_OGM1():
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
    plt.savefig("./homework/hw03/htc2022_orig.png")

    gd = GD(A, sino, x0)
    ogm = OGM1(A, sino, x0)

    res = []
    res.append(gd.leastSquares())
    res.append(gd.l2Norm())
    res.append(ogm.leastSquares())
    res.append(ogm.l2Norm())

    titles = ["GD-LeastSquares", "GD-Tikhonov", "OGM1-LeastSquares", "OGM1-Tikhonov"]
    score = []
    x = np.arange(2, 6)
    fig, ax = plt.subplots(2, 2)
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].imshow(res[i * 2 + j], cmap="gray")
            ax[i, j].set_title(titles[i * 2 + j])
            tifffile.imwrite(
                "./homework/hw03/htc2022_" + titles[i * 2 + j] + ".tif", res[i * 2 + j]
            )
            score.append(calculate_score(segment(res[i * 2 + j]), segment(ground)))
    fig.suptitle("Comparison of different methods")
    plt.tight_layout()
    plt.savefig("./homework/hw03/GD_OGM.png")

    plt.figure()
    plt.scatter(x, score)
    plt.xticks(x, titles)
    plt.ylim(0, 1)
    plt.title("Score of different methods")
    plt.savefig("./homework/hw03/GD_OGM_err.png")

    print(score)

    return


def test_Landweber():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full"
    )
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    lw = LW(A, sino, x0)
    sigma2 = lw.sigma2()

    mult = [1, 1e2, 1e4, 1e6]
    lam = [1 / (sigma2 * i) for i in mult]
    res = []
    score = []
    for l in lam:
        img = lw.solve(l)
        res.append(img)
        score.append(calculate_score(segment(img), segment(ground)))

    fig, ax = plt.subplots(2, 2)
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].imshow(res[i * 2 + j], cmap="gray")
            ax[i, j].set_title("lambda=1/(sigma^2*" + str(int(mult[i * 2 + j])) + ")")
            tifffile.imwrite(
                "./homework/hw03/htc2022_Landweber_" + str(mult[i * 2 + j]) + ".tif",
                res[i * 2 + j],
            )
    fig.suptitle("Landweber iteration")
    plt.tight_layout()
    plt.savefig("./homework/hw03/landweber.png")

    plt.figure()
    plt.scatter(lam, score)
    plt.ylim(0, 1)
    plt.xscale("log")
    plt.title("Score of different lambda")
    plt.tight_layout()
    plt.savefig("./homework/hw03/landweber_score.png")

    print(score)

    return


def test_CGD():
    sino, A = load_htc2022data(
        "/srv/ceph/share-all/aomip/htc2022_test_data/htc2022_07c_full"
    )
    x_shape = np.array([512, 512])
    x0 = np.zeros(x_shape)

    ground = tifffile.imread(
        "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_07c_recon.tif"
    )

    it = [1, 5, 10, 100]
    scores = []
    for i in it:
        img = CGD(A, sino, x0, nmax=i)
        scores.append("{:.4f}".format(calculate_score(segment(img), segment(ground))))

    print(scores)

    res = {}
    res["Gradient Descent"] = GD(A, sino, x0).leastSquares()
    res["OGM1"] = OGM1(A, sino, x0).leastSquares()
    res["Landweber"] = LW(A, sino, x0).solve()
    res["Conjugate Gradient Descent"] = CGD(A, sino, x0, nmax=1)

    l = list(res.items())
    fig, ax = plt.subplots(2, 2)
    score = {}
    for i in range(len(l)):
        ax[i // 2, i % 2].imshow(l[i][1], cmap="gray")
        ax[i // 2, i % 2].set_title(l[i][0])
        tifffile.imwrite("./homework/hw03/htc2022_" + l[i][0] + ".tif", l[i][1])
        score[l[i][0]] = "{:.4f}".format(
            calculate_score(segment(l[i][1]), segment(ground))
        )
    fig.suptitle("Comparison of different methods")
    plt.tight_layout()
    plt.savefig("./homework/hw03/CGD.png")

    print(score)

    return


def test_noise():
    ground = tifffile.imread("./homework/hw03/5.1.12.tiff")
    n = noise(ground)
    img = {}
    img["Gaussian"] = n.gaussian()
    img["Poisson"] = n.poisson()
    img["Salt and Pepper"] = n.salt_pepper()
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(ground, cmap="gray")
    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(img["Gaussian"], cmap="gray")
    ax[0, 1].set_title("Gaussian")
    ax[1, 0].imshow(img["Poisson"], cmap="gray")
    ax[1, 0].set_title("Poisson")
    ax[1, 1].imshow(img["Salt and Pepper"], cmap="gray")
    ax[1, 1].set_title("Salt and Pepper")
    fig.suptitle("Different types of noise")
    plt.tight_layout()
    plt.savefig("./homework/hw03/noise.png")
    x0 = np.zeros(ground.shape).flatten()
    res = {}
    nmax = 1e7
    for i in img.items():
        for beta in [1e-2, 1e-3, 1e-4, 1e-5]:
            l2Norm = lambda x: (x - i[1].flatten()) + beta * x
            res[i[0] + ", beta = {:.5f}".format(beta)] = gradDesc(
                l2Norm, x0, nmax=nmax
            ).reshape(ground.shape)
    score = {}
    for el in res.items():
        score[el[0]] = "{:.3f}".format(
            np.linalg.norm(el[1] - ground) / np.linalg.norm(ground)
        )
    print(score)
    ref_score = {}
    for el in img.items():
        ref_score[el[0]] = "{:.3f}".format(
            np.linalg.norm(el[1] - ground) / np.linalg.norm(ground)
        )
    print(ref_score)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img["Gaussian"], cmap="gray")
    ax[0].set_title("Gaussian noise")
    ax[1].imshow(res["Gaussian, beta = 0.01000"], cmap="gray")
    ax[1].set_title("Denoising, beta = 0.01000")
    plt.tight_layout()
    plt.savefig("./homework/hw03/denoise_gauss.png")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img["Poisson"], cmap="gray")
    ax[0].set_title("Poisson noise")
    ax[1].imshow(res["Poisson, beta = 0.01000"], cmap="gray")
    ax[1].set_title("Denoising, beta = 0.01000")
    plt.tight_layout()
    plt.savefig("./homework/hw03/denoise_poisson.png")
    return


def test_convolution():
    ground = tifffile.imread("./homework/hw03/5.1.12.tiff")
    ground[256:356, 256:356] = 255
    convoluted = GFconvolution(ground)
    deconvoluted = GFdeconvolution(convoluted)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ground, cmap="gray")
    ax[0].set_title("Original")
    ax[1].imshow(convoluted, cmap="gray")
    ax[1].set_title("Convoluted")
    ax[2].imshow(deconvoluted, cmap="gray")
    ax[2].set_title("Deconvoluted")
    plt.tight_layout()
    plt.savefig("./homework/hw03/convolution.png")

    gaussian = noise(convoluted).gaussian()
    deconvoluted = GFdeconvolution(gaussian)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(gaussian, cmap="gray")
    ax[0].set_title("Convoluted, with Gaussian noise")
    ax[1].imshow(deconvoluted, cmap="gray")
    ax[1].set_title("Deconvoluted")
    plt.tight_layout()
    plt.savefig("./homework/hw03/convolution_gaussian.png")

    salt_pepper = noise(convoluted).salt_pepper()
    deconvoluted = GFdeconvolution(salt_pepper)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(salt_pepper, cmap="gray")
    ax[0].set_title("Convoluted, with Salt and Pepper noise")
    ax[1].imshow(deconvoluted, cmap="gray")
    ax[1].set_title("Deconvoluted")
    plt.tight_layout()
    plt.savefig("./homework/hw03/convolution_salt_pepper.png")
    return


def test_deconvolution():
    ground = tifffile.imread("./homework/hw03/5.1.12.tiff")
    blurred = GFconvolution(ground)
    gauss = noise(blurred).gaussian()
    salt_pepper = noise(blurred).salt_pepper()
    x0 = np.ones(ground.shape) * 255
    nmax = 1e4
    res = {}
    for beta in [1e-2, 1e-3, 1e-4, 1e-5]:
        l2Norm = (
            lambda x: GFdeconvolution(
                GFconvolution(x.reshape(ground.shape)) - gauss
            ).flatten()
            + beta * x
        )
        res["Gauss noise, beta = {:.5f}".format(beta)] = gradDesc(l2Norm, x0, nmax=nmax)
    for beta in [1e-2, 1e-3, 1e-4, 1e-5]:
        l2Norm = (
            lambda x: GFdeconvolution(
                GFconvolution(x.reshape(ground.shape)) - salt_pepper
            ).flatten()
            + beta * x
        )
        res["Salt and Pepper noise, beta = {:.5f}".format(beta)] = gradDesc(
            l2Norm, x0, nmax=nmax
        )
    err = {}
    for el in res.items():
        err[el[0]] = "{:.3f}".format(
            np.linalg.norm(np.clip(el[1], 0, 255) - ground) / np.linalg.norm(ground)
        )
    print(err)
    ref = {}
    ref["Gauss noise"] = "{:.3f}".format(
        np.linalg.norm(gauss - ground) / np.linalg.norm(ground)
    )
    ref["Salt and Pepper noise"] = "{:.3f}".format(
        np.linalg.norm(salt_pepper - ground) / np.linalg.norm(ground)
    )
    print(ref)
    return


def test_all():
    test_OGM1()
    test_Landweber()
    test_CGD()
    test_noise()
    test_convolution()
    test_deconvolution()
    return


if __name__ == "__main__":
    test_deconvolution()
