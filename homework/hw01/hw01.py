from aomip import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.transform import iradon as iradon_sci


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
    I0 = findI0(image)
    transmission = getTransmission(image.copy(), I0)
    absorption = getAbsorption(image.copy(), I0)
    a_to_t = absorptionToTransmission(absorption)
    t_to_a = transmissionToAbsorption(transmission)
    fig = plt.figure()
    fig.suptitle("Negative log transform")
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(image, cmap="gray")
    fig.colorbar(im1, ax=ax1, location="left")
    ax1.set_title("Original image")
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(transmission, cmap="gray")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Transmission")
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(absorption, cmap="gray")
    fig.colorbar(im3, ax=ax3)
    ax3.set_title("Absorption")
    ax4 = fig.add_subplot(gs[2, 0])
    im4 = ax4.imshow(a_to_t, cmap="gray")
    fig.colorbar(im4, ax=ax4)
    ax4.set_title("Absorption to transmission")
    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(t_to_a, cmap="gray")
    fig.colorbar(im5, ax=ax5)
    ax5.set_title("Transmission to absorption")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/negativeLogTransform.png")
    plt.show()


def test_Padding():
    cwd = os.getcwd()
    image = mpimg.imread(cwd + "/homework/hw01/tubeV1/scan_000006.tif")
    image = vPad(image, 1)
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
    return


def test_FilteredBackProjection():
    cwd = os.getcwd()
    arc = 180
    angles = 20
    sino_shape = [420]
    phantom = shepp_logan([320, 320])
    sino = radon(phantom, sino_shape, np.linspace(0, arc, angles), 1000, 150)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom, cmap="gray")
    ax[0].set_title("Shepp Logan phantom")
    ax[1].imshow(sino, cmap="gray")
    ax[1].set_title("Sinogram")
    fig.suptitle("Forward projection")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/forwardProjection.png")
    plt.show()
    fig, ax = plt.subplots(3, 2)
    fig.suptitle("Back projection")
    img_sci_no = iradon_sci(sino, np.linspace(0, arc, angles), filter_name="shepp-logan")
    img_sci_ra = iradon_sci(sino, np.linspace(0, arc, angles), filter_name="ramp")
    img_sci_co = iradon_sci(sino, np.linspace(0, arc, angles), filter_name="cosine")
    img_no = iradon(sino, phantom.shape, np.linspace(0, arc, angles), 1000, 150, filter="shepp-logan")
    img_ra = iradon(sino, phantom.shape, np.linspace(0, arc, angles), 1000, 150, filter="ram-lak")
    img_co = iradon(sino, phantom.shape, np.linspace(0, arc, angles), 1000, 150, filter="cosine")
    ax[0][0].imshow(img_sci_no, cmap="gray")
    ax[0][0].set_title("Skimage, shepp-logan filter")
    ax[0][1].imshow(img_no, cmap="gray")
    ax[0][1].set_title("Shepp-logan filter")
    ax[1][0].imshow(img_sci_ra, cmap="gray")
    ax[1][0].set_title("Skimage, ramp filter")
    ax[1][1].imshow(img_ra, cmap="gray")
    ax[1][1].set_title("Ram-lak filter")
    ax[2][0].imshow(img_sci_co, cmap="gray")
    ax[2][0].set_title("Skimage, cosine filter")
    ax[2][1].imshow(img_co, cmap="gray")
    ax[2][1].set_title("Cosine filter")
    plt.tight_layout()
    plt.savefig(cwd + "/homework/hw01/backProjection.png")
    plt.show()

def test_all():
    test_flatFieldCorrection()
    test_binning()
    test_centerOfRotationCorrection()
    test_NegativeLogTransform()
    test_Padding()
    test_FilteredBackProjection()
    return


if __name__ == "__main__":
    test_all()
