from napari_clusters_plotter._plotter import PlotterWidget
from qtpy.QtCore import QSize
import numpy as np


class PhasorPlotterWidget(PlotterWidget):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.setMinimumSize(QSize(200, 200))

    def run(self,
            features,
            plot_x_axis_name,
            plot_y_axis_name,
            plot_cluster_name=None,
            redraw_cluster_image=True,
            ):
        super().run(features=features,
                    plot_x_axis_name=plot_x_axis_name,
                    plot_y_axis_name=plot_y_axis_name,
                    plot_cluster_name=plot_cluster_name,
                    redraw_cluster_image=redraw_cluster_image, )

        self.axes_limits()

    def axes_limits(self):
        self.add_phasor_circle(self.graphics_widget.axes)
        self.graphics_widget.axes.set_ylim([-1, 1])
        self.graphics_widget.axes.set_xlim([-1, 1])
        self.graphics_widget.draw_idle()

    def add_phasor_circle(self, ax):
        """
            Built the figure inner and outer circle and the 45 degrees lines in the plot
        :param ax: axis where to plot the phasor circle.
        :return: the axis with the added circle.
        """
        x1 = np.linspace(start=-1, stop=1, num=500)
        yp1 = lambda x1: np.sqrt(1 - x1 ** 2)
        yn1 = lambda x1: -np.sqrt(1 - x1 ** 2)
        x2 = np.linspace(start=-0.5, stop=0.5, num=500)
        yp2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
        yn2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
        x3 = np.linspace(start=-1, stop=1, num=30)
        x4 = np.linspace(start=-0.7, stop=0.7, num=30)
        ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
        ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
        ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
        ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
        ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
        ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
        ax.scatter(x4, x4, marker='_', color='darkgoldenrod')
        ax.scatter(x4, -x4, marker='_', color='darkgoldenrod')
        ax.annotate('0º', (1, 0), color='darkgoldenrod')
        ax.annotate('180º', (-1, 0), color='darkgoldenrod')
        ax.annotate('90º', (0, 1), color='darkgoldenrod')
        ax.annotate('270º', (0, -1), color='darkgoldenrod')
        ax.annotate('0.5', (0.42, 0.28), color='darkgoldenrod')
        ax.annotate('1', (0.8, 0.65), color='darkgoldenrod')
        return ax
