import numpy as np
from skimage.filters import median


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


def histogram_thresholding(dc, g, s, imin, imax=None):
    """
        Use this function to filter the background deleting, those pixels where the intensity value is under ic.
    :param dc: ndarray. HSI stack average intensity image.
    :param g: ndarray. G image.
    :param s: ndarray. S image.
    :param imin: Type integer. Minimum cutoff intensity value.
    :param imax: Type integer. Maximum cutoff intensity value.
    :return: x, y. Arrays contain the G and S phasor coordinates.
    """
    if dc.any():
        if g.any():
            if s.any():
                if isinstance(imin, int):
                    aux = np.concatenate(np.where(dc > imin, dc, np.zeros(dc.shape)))
                    g = np.concatenate(g)
                    s = np.concatenate(s)
                    if imax:
                        if isinstance(imax, int):
                            aux2 = np.concatenate(np.where(dc < imax, np.ones(dc.shape), np.zeros(dc.shape)))
                            aux = aux * aux2
                            x = np.delete(g, np.where(aux == 0))
                            y = np.delete(s, np.where(aux == 0))
                        else:
                            raise ValueError("imax value is not an integer")
                    else:
                        x = np.delete(g, np.where(aux == 0))
                        y = np.delete(s, np.where(aux == 0))
                    return x, y, aux
                else:
                    raise ValueError("imin value is not an integer")
            else:
                raise ValueError("Empty s array")
        else:
            raise ValueError("Empty g array")
    else:
        raise ValueError("Empty dc array")


def cursor_mask(dc, g, s, center, Ro, ncomp=5):
    """
        Create a matrix to see if a pixels is into the circle, using circle equation
    so the negative values of Mi means that the pixel belong to the circle and multiply
    aux1 to set zero where the avg image is under ic value
    :param ncomp: number of cursors to be used in the phasor, and the pseudocolor image.
    :param dc: ndarray. Intensity image.
    :param g:  ndarray. G image.
    :param s:  ndarray. S image.
    :param ic: intensity cut umbral. Default 0
    :param Ro: circle radius.
    :param center: ndarray containing the center coordinate of each circle.
    :return: rgba pseudocolored image.
    """
    img = np.zeros([dc.shape[0], dc.shape[1], 3])
    ccolor = [[128, 0, 128], [0, 0, 1], [0, 1, 0], [255, 255, 0], [1, 0, 0]]  # colors are v, b, g, y, r
    for i in range(ncomp):
        M = ((g - center[i][0]) ** 2 + (s - center[i][1]) ** 2 - Ro ** 2)
        indices = np.where(M < 0)
        img[indices[0], indices[1], :3] = ccolor[i]
    return img
