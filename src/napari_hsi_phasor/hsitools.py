import numpy as np
from skimage.filters import median
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari.types


def phasor(image_stack, harmonic=1):
    """
        This function computes the average intensity image, the G and S coordinates, modulation and phase.

    :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from numpy library.
    :param harmonic: int. The number of the harmonic where the phasor is calculated.
                            harmonic range: [1, num of channels - 1]
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    if image_stack.any():
        if isinstance(harmonic, int) and 0 < harmonic < len(image_stack):
            data = np.fft.fft(image_stack, axis=0, norm='ortho')
            dc = data[0].real
            dc = np.where(dc != 0, dc, int(np.mean(dc, dtype=np.float64)))  # change the zeros to the img average
            g = data[harmonic].real
            g /= dc
            s = data[harmonic].imag
            s /= -dc
            avg = np.mean(image_stack, axis=0, dtype=np.float64)
        else:
            raise ValueError("harmonic indices is not integer or slice or harmonic out of range\n harmonic range: [1, "
                             "num of channels - 1]")
        return avg, g, s
    else:
        raise ValueError("Image stack data is not an array")


def median_filter(im, n):
    """
        Apply median filter to an image from the skimage.filter module
    :param im: image to be filtered.
    :param n: nth times to filter im
    """
    if im.any():
        im_aux = np.copy(im)
        for i in range(n):
            im_aux = median(im_aux)
        return im_aux
    else:
        raise ValueError("Image stack data is not an array")

