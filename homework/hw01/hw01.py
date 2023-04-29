from aomip import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np


def test_flatFieldCorrection():
    cwd = os.getcwd()
    dark_frame = mpimg.imread(cwd + "/homework/hw01/tubeV1/di000000.tif")
    flat_field = mpimg.imread(cwd + "/homework/hw01/tubeV1/io000000.tif")
    fig, ax = plt.subplots(5, 4, layout="compressed")
    fig.suptitle("Flat-field correction")
    for i in range(10):
        image = mpimg.imread(
            cwd + "/homework/hw01/tubeV1/scan_000" + str(i) + str(i) + "6.tif"
        )
        if i < 5:
            ax[i][0].imshow(image, cmap="gray")
            ax[i][0].axis("off")
            ax[i][1].imshow(
                flatFieldCorrection(image, dark_frame, flat_field), cmap="gray"
            )
            ax[i][1].axis("off")
        else:
            ax[i - 5][2].imshow(image, cmap="gray")
            ax[i - 5][2].axis("off")
            ax[i - 5][3].imshow(
                flatFieldCorrection(image, dark_frame, flat_field), cmap="gray"
            )
            ax[i - 5][3].axis("off")
    ax[0][0].set_title("Original image")
    ax[0][1].set_title("Corrected image")
    ax[0][2].set_title("Original image")
    ax[0][3].set_title("Corrected image")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/flatFieldCorrection.png")
    plt.show()
    return


def test_binning():
    x = np.array([[1, 2, 3, 4], [80, 70, 6, 5], [9, 10, 11, 12], [16, 90, 14, 100]])
    x_bin = binning(x)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 14, 13, 12, 11, 10, 9])
    y_bin = binning(y)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(x)
    ax[0][1].imshow(x_bin)
    ax[1][0].imshow(np.reshape(y, (len(y), 1)))
    ax[1][0].xaxis.set_visible(False)
    ax[1][1].imshow(np.reshape(y_bin, (len(y_bin), 1)))
    ax[1][1].xaxis.set_visible(False)
    ax[0][0].set_title("Original array")
    ax[0][1].set_title("Binned array")
    ax[1][0].set_title("Original array")
    ax[1][1].set_title("Binned array")
    fig.suptitle("Binning on 2D and 1D arrays")
    cwd = os.getcwd()
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/binningRandom.png")
    plt.show()
    image = mpimg.imread(cwd + "/homework/hw01/tubeV1/scan_000006.tif")
    image_bin = binning(image)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Binning on a 2D image")
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original image")
    ax[1].imshow(image_bin, cmap="gray")
    ax[1].set_title("Binned image")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/binningImage.png")
    plt.show()
    return


def test_centerOfRotationCorrection():
    cwd = os.getcwd()
    image = mpimg.imread(cwd + "/homework/hw01/tubeV1/scan_000006.tif")
    image_20_0 = centerOfRotationCorrection(image, 20, 0)
    image_200_0 = centerOfRotationCorrection(image, 200, 0)
    image_20_1 = centerOfRotationCorrection(image, 20, 1)
    image_200_1 = centerOfRotationCorrection(image, 200, 1)
    fig, ax = plt.subplots(2, 3)
    fig.suptitle("Center of rotation correction")
    ax[0][0].imshow(image, cmap="gray")
    ax[0][0].set_title("Original image")
    ax[0][1].imshow(image_20_0, cmap="gray")
    ax[0][1].set_title("20px offset, axis 0")
    ax[0][2].imshow(image_200_0, cmap="gray")
    ax[0][2].set_title("200px offset, axis 0")
    ax[1][0].imshow(image, cmap="gray")
    ax[1][0].set_title("Original image")
    ax[1][1].imshow(image_20_1, cmap="gray")
    ax[1][1].set_title("20px offset, axis 1")
    ax[1][2].imshow(image_200_1, cmap="gray")
    ax[1][2].set_title("200px offset, axis 1")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/centerOfRotationCorrection.png")
    plt.show()
    return


def test_NegativeLogTransform():
    cwd = os.getcwd()
    image = mpimg.imread(cwd + "/homework/hw01/tubeV1/scan_000006.tif")
    image = flatFieldCorrection(
        image,
        mpimg.imread(cwd + "/homework/hw01/tubeV1/di000000.tif"),
        mpimg.imread(cwd + "/homework/hw01/tubeV1/io000000.tif"),
    )
    I0 = findI0(image)
    transmission = transmissionToAbsorption(image.copy(), I0)
    absorption = absorptionToTransmission(image.copy(), I0)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image, cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(transmission, cmap="gray")
    ax[1].axis("off")
    ax[2].imshow(absorption, cmap="gray")
    ax[2].axis("off")
    # plt.savefig(cwd+"/homework/hw01/originalImage.png")
    plt.show()
    return


def test_Padding():
    cwd = os.getcwd()
    image = mpimg.imread(cwd + "/homework/hw01/tubeV1/scan_000006.tif")
    v_pad_image = vPad(image, 100)
    h_pad_image = hPad(image, 100)
    b_pad_image = hPad(v_pad_image, 100)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(image, cmap="gray")
    ax[0][1].imshow(b_pad_image, cmap="gray")
    ax[1][0].imshow(v_pad_image, cmap="gray")
    ax[1][1].imshow(h_pad_image, cmap="gray")
    ax[0][0].set_title("Original image")
    ax[0][1].set_title("Both padding")
    ax[1][0].set_title("Vertical padding")
    ax[1][1].set_title("Horizontal padding")
    fig.suptitle("Padding")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/padding.png")
    plt.show()


if __name__ == "__main__":
    # test_flatFieldCorrection()
    # test_binning()
    # test_centerOfRotationCorrection()
    # test_NegativeLogTransform()
    test_Padding()
