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

    def toggle_tile_widget(event):
        widget.tile.visible = event
    # Connect events
    widget.tile.changed.connect(toggle_tile_widget)
    # Intial visibility states
    widget.tile.visible = False



@magic_factory(widget_init=connect_events)
def phasor_plot(image_layer: "napari.layers.Image",
                harmonic: int = 1,
                threshold: int = 0,
                apply_median: bool = False,
                median_n: int = 1,
                tile: bool = False,
                vertical_dim: int = 1024,
                horizontal_dim: int = 1024,
                vertical_im: int = 1,
                horizontal_im: int = 1,
                v_overlapping_per: float = 0.05,
                h_overlapping_per: float = 0.05,
                store_dir: bool = False, # set true for bidirectional storing 
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
    import dask.array as da

    from napari_hsi_phasor.hsitools import phasor, median_filter, tilephasor, stitching
    from napari_hsi_phasor._plotter import PhasorPlotterWidget
    from skimage.segmentation import relabel_sequential

    image = image_layer.data

    if tile:
        dc, g, s = tilephasor(image, vertical_dim, horizontal_dim, harmonic=harmonic)
        dc = stitching(dc, vertical_im, horizontal_im, h_overlapping_per, v_overlapping_per, store_dir)
        g = stitching(g, vertical_im, horizontal_im, h_overlapping_per, v_overlapping_per, store_dir)
        s = stitching(s, vertical_im, horizontal_im, h_overlapping_per, v_overlapping_per, store_dir)

    else:
        dc, g, s = phasor(image, harmonic=harmonic)

    if apply_median:
        g = median_filter(g, median_n)
        s = median_filter(s, median_n)

    dc = dc[np.newaxis, np.newaxis, :, :]
    g = g[np.newaxis, np.newaxis, :, :]
    s = s[np.newaxis, np.newaxis, :, :]

    space_mask = dc > threshold
    label_image = np.arange(np.prod(dc.shape)).reshape(dc.shape) + 1
    label_image[~space_mask] = 0
    label_image = relabel_sequential(label_image)[0]
    label_column = np.ravel(label_image[space_mask])

    g_flat_masked = np.ravel(g[space_mask])
    s_flat_masked = np.ravel(s[space_mask])
    if isinstance(g, da.Array):
        g_flat_masked.compute_chunk_sizes()
        s_flat_masked.compute_chunk_sizes()

    phasor_components = pd.DataFrame({
        'label': np.ravel(label_image[space_mask]),
        'G': g_flat_masked,
        'S': s_flat_masked})
    table = phasor_components

    frame = np.arange(dc.shape[0])
    frame = np.repeat(frame, np.prod(dc.shape[1:]))
    table['frame'] = frame[space_mask.ravel()]

    # The layer has to be created here so the plotter can be filled properly
    # below. Overwrite layer if it already exists.
    for layer in napari_viewer.layers:
        if (isinstance(layer, Labels)) & (layer.name == 'Labelled_pixels_from_' + image_layer.name):
            labels_layer = layer
            labels_layer.data = label_image
            labels_layer.features = table
            break
    else:
        labels_layer = napari_viewer.add_labels(label_image,
                                                name='Labelled_pixels_from_' + image_layer.name,
                                                features=table,
                                                scale=image_layer.scale[1:],
                                                visible=False)

    # Check if plotter was alrerady added to dock_widgets
    # TO DO: avoid using private method access to napari_viewer.window._dock_widgets (will be deprecated)
    dock_widgets_names = [key for key,
                                  value in napari_viewer.window._dock_widgets.items()]
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
    plotter_widget.axes_limits()
    return
