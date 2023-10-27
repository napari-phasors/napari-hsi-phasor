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


def tilephasor(image_stack, dimx, dimy, harmonic=1):
    """
        This function compute the fft and calculate the phasor for an stack containing many tiles
        of images.
    :param dimy: images horizontal dimension
    :param dimx: images vertical dimension
    :param image_stack: image stack containing the n lambda channels
    :param harmonic: The nth harmonic of the phasor. Type int.
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    if image_stack.any():
        if isinstance(harmonic, int) and 0 < harmonic < len(image_stack):
            dc = np.zeros([len(image_stack), dimx, dimy])
            g = np.zeros([len(image_stack), dimx, dimy])
            s = np.zeros([len(image_stack), dimx, dimy])
            for i in range(len(image_stack)):
                dc[i], g[i], s[i] = phasor(image_stack[i], harmonic=harmonic)
            return dc, g, s
        else:
            raise ValueError("harmonic indices is not integer or slice or harmonic out of range\n harmonic range: [1, "
                             "num of channels - 1]")
    else:
        raise ValueError("Image stack data is an empty array")


def stitching(im, m, n, hper=0.05, vper=0.05, bidirectional=False):
    """
        Stitches a stack image from mxn images create an m x n only image.
    :param im: image stack to be concatenated, containing mxn images.
    :param m: number of vertical images
    :param n: number of horizontal images
    :param hper: horizontal percentage of overlap
    :param vper: vertical percentage of overlap
    :param bidirectional: Optional, set true if the image tile are bidirectional array
    :return: concatenated image
    """
    if im.any():
        if isinstance(m, int):
            if isinstance(n, int):
                d = im.shape[1]
                aux = np.zeros([d * m, d * n])  # store the concatenated image
                # Horizontal concatenate
                i = 0
                j = 0
                while j < m * n:
                    if bidirectional and ((j / n) % 2 == 1):
                        aux[i * d: i * d + d, 0:d] = im[j + (n - 1)][0:, 0:d]  # store the first image horizontally
                    else:
                        aux[i * d: i * d + d, 0:d] = im[j][0:, 0:d]  # store the first image horizontally
                    k = 1
                    acum = 0
                    if bidirectional and ((j / n) % 2 == 1):
                        while k < n:
                            ind1 = round(((1 - vper) + acum) * d)
                            ind2 = round(ind1 + vper * d)
                            ind3 = round(ind2 + (1 - vper) * d)
                            aux[i * d:i * d + d, ind1:ind2] = (aux[i * d:i * d + d, ind1:ind2] + im[j + (n - k - 1)][0:,
                                                                                                 0:round(vper * d)]) / 2
                            aux[i * d:i * d + d, ind2:ind3] = im[j + (n - k - 1)][0:, round(vper * d):d]
                            acum = (1 - vper) + acum
                            k = k + 1
                    else:
                        while k < n:
                            ind1 = round(((1 - vper) + acum) * d)
                            ind2 = round(ind1 + vper * d)
                            ind3 = round(ind2 + (1 - vper) * d)
                            aux[i * d:i * d + d, ind1:ind2] = (aux[i * d:i * d + d, ind1:ind2] + im[j + k][0:,
                                                                                                 0:round(vper * d)]) / 2
                            aux[i * d:i * d + d, ind2:ind3] = im[j + k][0:, round(vper * d):d]
                            acum = (1 - vper) + acum
                            k = k + 1
                    i = i + 1
                    j = j + n

                # Vertical concatenate
                img = np.zeros([round(d * (m - hper * (m - 1))), round(d * (n - hper * (n - 1)))])
                img[0:d, 0:] = aux[0:d, 0:img.shape[1]]
                k = 1
                while k < m:
                    #  indices de la matrix aux para promediar las intersecciones
                    ind1 = round(d * (k - hper))
                    ind2 = round(d * k)
                    ind3 = round(d * (k + hper))
                    ind4 = round(d * (k + 1))
                    #  indices de la nueva matriz donde se almacena la imagen final
                    i1 = round(k * d * (1 - hper))
                    i2 = round(i1 + d * hper)
                    i3 = round(i2 + d * (1 - hper))

                    img[i1:i2, 0:] = (aux[ind1:ind2, 0:img.shape[1]] + aux[ind2:ind3, 0:img.shape[1]]) / 2
                    img[i2:i3, 0:] = aux[ind3:ind4, 0:img.shape[1]]
                    k = k + 1

                return img
            else:
                raise ValueError("n value is not an integer")
        else:
            raise ValueError("m value is not an integer")
    else:
        raise ValueError("Empty image array")