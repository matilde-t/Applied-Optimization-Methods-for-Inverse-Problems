import numpy as np


def GFconvolution(img, sigma=0.2):
    H = np.linspace(-1, 1, img.shape[0])

    H = np.exp(-(H**2) / (2 * sigma**2))

    h = np.tile(H, (img.shape[1], 1)).T

    fftimg = np.fft.fft(img, axis=0)
    projection = np.fft.fftshift(fftimg, axes=1) * np.fft.fftshift(h, axes=0)
    fsimg = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))

    return np.clip(fsimg, 0, 255)


def GFdeconvolution(img, sigma=0.2):
    H = np.linspace(-1, 1, img.shape[0])

    H = np.exp(-(H**2) / (2 * sigma**2))

    h = np.tile(H, (img.shape[1], 1)).T

    fftimg = np.fft.fft(img, axis=0)
    projection = np.fft.fftshift(fftimg, axes=1) / np.fft.fftshift(h, axes=0)
    fsimg = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))

    return np.clip(fsimg, 0, 255)
