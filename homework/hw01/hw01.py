from aomip import flatFieldCorrection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def test_flatFieldCorrection():
    cwd=os.getcwd()
    dark_frame = mpimg.imread(cwd+"/homework/hw01/tubeV1/di000000.tif")
    flat_field = mpimg.imread(cwd+"/homework/hw01/tubeV1/io000000.tif")
    fig, ax = plt.subplots(5, 4, layout="compressed")
    fig.suptitle("Flat field correction")
    for i in range(10):
        image = mpimg.imread(cwd+"/homework/hw01/tubeV1/scan_000"+str(i)+str(i)+"6.tif")
        if i<5:
            ax[i][0].imshow(image, cmap="gray")
            ax[i][0].axis("off")
            ax[i][1].imshow(flatFieldCorrection(image, dark_frame, flat_field), cmap="gray")
            ax[i][1].axis("off")
        else:
            ax[i-5][2].imshow(image, cmap="gray")
            ax[i-5][2].axis("off")
            ax[i-5][3].imshow(flatFieldCorrection(image, dark_frame, flat_field), cmap="gray")
            ax[i-5][3].axis("off")
    ax[0][0].set_title("Original image")
    ax[0][1].set_title("Corrected image")
    ax[0][2].set_title("Original image")
    ax[0][3].set_title("Corrected image")
    plt.tight_layout()
    plt.savefig(cwd+"/homework/hw01/flatFieldCorrection.png")
    plt.show()
    
    

if __name__ == "__main__":
    test_flatFieldCorrection()