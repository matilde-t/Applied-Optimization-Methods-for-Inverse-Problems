import aomip
import numpy as np
import matplotlib.pyplot as plt
import os 
import tifffile

def test_Slicing():
    cwd = os.getcwd()
    path = cwd + '/homework/hw02/sea/'
    img = []
    for image in os.listdir(path):
        tmp = aomip.centerOfRotationCorrection(tifffile.imread(path + image), -4, 1) # correction as suggested in the dataset
        img.append(tmp)
    sino = aomip.slice(img, 5)
    plt.imshow(sino, cmap='gray')
    plt.show()
    
    

if __name__ == '__main__':
    test_Slicing()