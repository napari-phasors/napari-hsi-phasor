import numpy as np


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
            md = np.sqrt(g ** 2 + s ** 2)
            ph = np.angle(data[harmonic], deg=True)
            avg = np.mean(image_stack, axis=0, dtype=np.float64)
        else:
            raise ValueError("harmonic indices is not integer or slice or harmonic out of range\n harmonic range: [1, "
                             "num of channels - 1]")
        return avg, g, s, md, ph
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
                    aux1 = np.concatenate(np.where(dc > imin, dc, np.zeros(dc.shape)))
                    g = np.concatenate(g)
                    s = np.concatenate(s)
                    if imax:
                        if isinstance(imax, int):
                            aux2 = np.concatenate(np.where(dc < imax, np.ones(dc.shape), np.zeros(dc.shape)))
                            aux = aux1 * aux2
                            x = np.delete(g, np.where(aux == 0))
                            y = np.delete(s, np.where(aux == 0))
                        else:
                            raise ValueError("imax value is not an integer")
                    else:
                        x = np.delete(g, np.where(aux1 == 0))
                        y = np.delete(s, np.where(aux1 == 0))
                    return x, y
                else:
                    raise ValueError("imin value is not an integer")
            else:
                raise ValueError("Empty s array")
        else:
            raise ValueError("Empty g array")
    else:
        raise ValueError("Empty dc array")


def imthreshold(im, imin, imax=None):
    """
    :param im: image to be thresholded
    :param imin: left intensity value threshold
    :param imax: right intensity value threshold. It is None there is no superior cutoff intensity
    :return: image threshold
    """
    if im.any():
        if isinstance(imin, int):
            imt1 = np.where(im > imin, im, np.zeros(im.shape))
            if isinstance(imax, int):
                imt2 = np.where(im < imax, im, np.zeros(im.shape))
                imt = imt1 * imt2
                return imt
            elif imt1.any():
                return imt1
            else:
                raise ValueError("imax value is not an integer")
        else:
            raise ValueError("imin value is not an integer")
    else:
        raise ValueError("Empty image array")
