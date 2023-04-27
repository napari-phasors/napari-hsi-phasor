"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import numpy as np

if TYPE_CHECKING:
    import napari


@magic_factory()
def phasor(image_stack: "napari.layers.Image", harmonic: int) -> "napari.types.LayerDataTuple":
    if isinstance(harmonic, int) and 0 < harmonic < len(image_stack):
        data = np.fft.fft(image_stack, axis=0, norm='ortho')
        dc = data[0].real
        dc = np.where(dc != 0, dc, int(np.mean(dc, dtype=np.float64)))
        g = data[harmonic].real
        g /= dc
        s = data[harmonic].imag
        s /= -dc
        md = np.sqrt(g ** 2 + s ** 2)
        ph = np.angle(data[harmonic], deg=True)
        avg = np.mean(image_stack, axis=0, dtype=np.float64)
        return np.asarray([avg, g, s, md, ph])
