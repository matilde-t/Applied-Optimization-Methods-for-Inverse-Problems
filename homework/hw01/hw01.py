from aomip import flatFieldCorrection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def test_flatFieldCorrection():
    cwd=os.getcwd()
    image = mpimg.imread(cwd+"/homework/hw01/walnut/20201111_walnut_0701.tif")
    plt.imshow(image)
    plt.show()
    

if __name__ == "__main__":
    test_flatFieldCorrection()