from typing import TYPE_CHECKING
from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


def connect_events(widget):
    """
    Connect widget events to make some visible/invisible depending on others
    """

    def toggle_median_n_widget(event):
        widget.median_n.visible = event

    # Connect events
    widget.apply_median.changed.connect(toggle_median_n_widget)
    # Intial visibility states
    widget.median_n.visible = False


@magic_factory(widget_init=connect_events)
def phasor_plot(image_layer: "napari.layers.Image",
                harmonic: int = 1,
                threshold: int = 0,
                apply_median: bool = False,
                median_n: int = 1,
                Ro: float = 0.5,
                napari_viewer: "napari.Viewer" = None) -> None:
    """Calculate phasor components from HSI image and plot them.
    Parameters
    ----------
    image_layer : napari.layers.Image
        napari image layer with HSI data.
    harmonic : int, optional
        the harmonic to display in the phasor plot, by default 1
    threshold : int, optional
        pixels with summed intensity below this threshold will be discarded, by default 0
    apply_median : bool, optional
        apply median filter to image before phasor calculation, by default False (median_n is ignored)
    median_n : int, optional
        number of iterations of median filter, by default 1
    napari_viewer : napari.Viewer, optional
        napari viewer instance, by default None
    """

    import numpy as np
    import pandas as pd
    from napari.layers import Labels

    from napari_hsi_phasor.hsitools import phasor, median_filter, histogram_thresholding, cursor_mask
    from napari_hsi_phasor._plotter import PhasorPlotterWidget

    image = image_layer.data
    g, s, dc = phasor(image, harmonic=harmonic)

    if apply_median:
        g = median_filter(g, median_n)
        s = median_filter(s, median_n)

    x, y, _ = histogram_thresholding(dc, g, s, imin=threshold)
    # mask = cursor_mask(dc, g, s, center, Ro, ncomp=5) todo: the mask need to take the (g, s) coordinate of the circle center

    phasor_components = pd.DataFrame({'label': dc, 'G': x, 'S': y}) # todo: revise the label because it must be the same size as g and s
    table = phasor_components
    # Build frame column
    frame = np.arange(dc.shape[0])
    frame = np.repeat(frame, np.prod(dc.shape[1:]))
    # table['frame'] = frame[mask.ravel()]

    # The layer has to be created here so the plotter can be filled properly
    # below. Overwrite layer if it already exists.
    for layer in napari_viewer.layers:
        if (isinstance(layer, Labels)) & (layer.name == 'Labelled_pixels_from_' + image_layer.name):
            labels_layer = layer
            labels_layer.data = dc
            # labels_layer.features = table
            break
    else:
        labels_layer = napari_viewer.add_labels(dc,
                                                name='Labelled_pixels_from_' + image_layer.name,
                                                # features=table,
                                                scale=image_layer.scale[1:],
                                                visible=False)

    # Check if plotter was already added to dock_widgets
    dock_widgets_names = [key for key, value in napari_viewer.window.dock_widgets.items()]
    if 'Phasor Plotter Widget (napari-hsi-phasor)' not in dock_widgets_names:
        plotter_widget = PhasorPlotterWidget(napari_viewer)
        napari_viewer.window.add_dock_widget(
            plotter_widget, name='Phasor Plotter Widget (napari-hsi-phasor)')
    else:
        widgets = napari_viewer.window._dock_widgets['Phasor Plotter Widget (napari-hsi-phasor)']
        plotter_widget = widgets.findChild(PhasorPlotterWidget)

    # Get labels layer with labelled pixels (labels)
    plotter_widget.labels_select.value = [
        choice for choice in plotter_widget.labels_select.choices if choice.name.startswith("Labelled_pixels")][0]
    # Set G and S as features to plot (update_axes_list method clears Comboboxes)
    plotter_widget.plot_x_axis.setCurrentIndex(1)
    plotter_widget.plot_y_axis.setCurrentIndex(2)
    plotter_widget.plotting_type.setCurrentIndex(1)

    # Show parent (PlotterWidget) so that run function can run properly
    plotter_widget.parent().show()
    # Disconnect selector to reset collection of points in plotter
    # (it gets reconnected when 'run' method is run)
    plotter_widget.graphics_widget.selector.disconnect()
    plotter_widget.run(labels_layer.features,
                       plotter_widget.plot_x_axis.currentText(),
                       plotter_widget.plot_y_axis.currentText())
    plotter_widget.redefine_axes_limits(ensure_full_semi_circle_displayed=True)
    return
