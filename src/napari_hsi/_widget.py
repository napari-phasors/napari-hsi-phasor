from typing import TYPE_CHECKING
from magicgui import magic_factory

from napari_clusters_plotter._plotter import PlotterWidget
from qtpy.QtCore import QSize
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget


def add_phasor_circle(ax):
    """
    Generate FLIM universal semi-circle plot
    """
    import numpy as np
    angles = np.linspace(0, np.pi, 180)
    x = (np.cos(angles) + 1) / 2
    y = np.sin(angles) / 2
    ax.plot(x, y, 'yellow', alpha=0.3)
    return ax


def add_tau_lines(ax, tau_list, frequency):
    import numpy as np
    if not isinstance(tau_list, list):
        tau_list = [tau_list]
    frequency = frequency * 1E6  # MHz to Hz
    w = 2 * np.pi * frequency  # Hz to radians/s
    for tau in tau_list:
        tau = tau * 1E-9  # nanoseconds to seconds
        g = 1 / (1 + ((w * tau)**2))
        s = (w * tau) / (1 + ((w * tau)**2))
        dot, = ax.plot(g, s, marker='o', mfc='none')
        array = np.linspace(0, g, 50)
        y = (array * s / g)
        ax.plot(array, y, color=dot.get_color())


def add_2d_histogram(ax, x, y):
    import matplotlib.pyplot as plt
    output = ax.hist2d(
        x=x,
        y=y,
        bins=10,
        cmap='jet',
        norm='log',
        alpha=0.5
    )
    return ax


class PhasorPlotterWidget(PlotterWidget):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.setMinimumSize(QSize(100, 300))

    def run(self,
            features,
            plot_x_axis_name,
            plot_y_axis_name,
            plot_cluster_name=None,
            redraw_cluster_image=True,):
        super().run(features=features,
                    plot_x_axis_name=plot_x_axis_name,
                    plot_y_axis_name=plot_y_axis_name,
                    plot_cluster_name=plot_cluster_name,
                    redraw_cluster_image=redraw_cluster_image,)
        add_phasor_circle(self.graphics_widget.axes)
        self.graphics_widget.draw()



if TYPE_CHECKING:
    import napari
import napari.layers



@magic_factory()
def phasor(image_stack: "napari.layers.Image",
           harmonic: int = 1,
           filt: int = 0,
           icut: int = 0,
           v2: int = 0,
           napari_viewer: "napari.Viewer" = None) -> None:

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

    # Check if plotter was alrerady added to dock_widgets
    # TO DO: avoid using private method access to napari_viewer.window._dock_widgets (will be deprecated)
    dock_widgets_names = [key for key,
                                  value in napari_viewer.window._dock_widgets.items()]
    if 'Plotter Widget' not in dock_widgets_names:
        plotter_widget = PhasorPlotterWidget(napari_viewer)
        napari_viewer.window.add_dock_widget(
            plotter_widget, name='Plotter Widget')
    else:
        widgets = napari_viewer.window._dock_widgets['Plotter Widget']
        plotter_widget = widgets.findChild(PhasorPlotterWidget)


    return x, y


