from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari
import napari.layers


@magic_factory()
def phasor(image_stack: "napari.layers.Image",
           harmonic: int = 1,
           filt: int = 0,
           icut: int = 0) -> None:
    import numpy as np
    from skimage.filters import median

    image = image_stack.data

    data = np.fft.fft(image, axis=0, norm='ortho')
    dc = data[0].real
    dc = np.where(dc != 0, dc, int(np.mean(dc, dtype=np.float64)))  # change the zeros to the img average
    g = data[harmonic].real
    g /= dc
    s = data[harmonic].imag
    s /= -dc
    avg = np.mean(image, axis=0, dtype=np.float64)

    if filt > 0:
        for i in range(filt):
            avg = median(avg)
            g = median(g)
            s = median(s)

    aux = np.concatenate(np.where(dc > icut, dc, np.zeros(dc.shape)))
    x = np.delete(np.concatenate(g), np.where(aux == 0))
    y = np.delete(np.concatenate(s), np.where(aux == 0))

    return
