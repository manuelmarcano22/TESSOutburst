# Import the optional Bokeh dependency, or print a friendly error otherwise.
import bokeh  # Import bokeh first so we get an ImportError we can catch
from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import LogColorMapper, Selection, Slider, RangeSlider, \
    Span, ColorBar, LogTicker, Range1d
from bokeh.layouts import layout, Spacer
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Button, Div
from bokeh.models.formatters import PrintfTickFormatter

import numpy as np


def prepare_lightcurve_datasource(lc):
    """Prepare a bokeh ColumnDataSource object for tool tips.
    Parameters
    ----------
    lc : LightCurve object
        The light curve to be shown.
    Returns
    -------
    lc_source : bokeh.plotting.ColumnDataSource
    """
    # Convert time into human readable strings, breaks with NaN time
    # See https://github.com/KeplerGO/lightkurve/issues/116
    if (lc.time == lc.time).all():
        human_time = lc.astropy_time.isot
    else:
        human_time = [' '] * len(lc.flux)

    # Convert binary quality numbers into human readable strings
    qual_strings = []
    for bitmask in lc.quality:
        flag_str_list = KeplerQualityFlags.decode(bitmask)
        if len(flag_str_list) == 0:
            qual_strings.append(' ')
        if len(flag_str_list) == 1:
            qual_strings.append(flag_str_list[0])
        if len(flag_str_list) > 1:
            qual_strings.append("; ".join(flag_str_list))

    lc_source = ColumnDataSource(data=dict(
                                 time=lc.time,
                                 time_iso=human_time,
                                 flux=lc.flux,
                                 cadence=lc.cadenceno,
                                 quality_code=lc.quality,
                                 quality=np.array(qual_strings)))
    return lc_source


def prepare_tpf_datasource(tpf, aperture_mask):
    """Prepare a bokeh DataSource object for selection glyphs
    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to be shown.
    aperture_mask : boolean numpy array
        The Aperture mask applied at the startup of interact
    Returns
    -------
    tpf_source : bokeh.plotting.ColumnDataSource
        Bokeh object to be shown.
    """
    npix = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, npix, 1).reshape(tpf.flux[0].shape)
    xx = tpf.column + np.arange(tpf.shape[2])
    yy = tpf.row + np.arange(tpf.shape[1])
    xa, ya = np.meshgrid(xx, yy)
    preselected = Selection()
    preselected.indices = pixel_index_array[aperture_mask].reshape(-1).tolist()
    tpf_source = ColumnDataSource(data=dict(xx=xa+0.5, yy=ya+0.5),
                                  selected=preselected)
    return tpf_source



def make_tpf_figure_elements(tpf, tpf_source, pedestal=None, fiducial_frame=None,
                             plot_width=370, plot_height=340):
    """Returns the lightcurve figure elements.
    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to show.
    tpf_source : bokeh.plotting.ColumnDataSource
        TPF data source.
    pedestal: float
        A scalar value to be added to the TPF flux values, often to avoid
        taking the log of a negative number in colorbars.
        Defaults to `-min(tpf.flux) + 1`
    fiducial_frame: int
        The tpf slice to start with by default, it is assumed the WCS
        is exact for this frame.
    Returns
    -------
    fig, stretch_slider : bokeh.plotting.figure.Figure, RangeSlider
    """
    if pedestal is None:
        pedestal = -np.nanmin(tpf.flux) + 1

    if tpf.mission in ['Kepler', 'K2']:
        title = 'Pixel data (CCD {}.{})'.format(tpf.module, tpf.output)
    elif tpf.mission == 'TESS':
        title = 'Pixel data (Camera {}.{})'.format(tpf.camera, tpf.ccd)
    else:
        title = "Pixel data"

    fig = figure(plot_width=plot_width, plot_height=plot_height,
                 x_range=(tpf.column, tpf.column+tpf.shape[2]),
                 y_range=(tpf.row, tpf.row+tpf.shape[1]),
                 title=title, tools='tap,box_select,wheel_zoom,reset',
                 toolbar_location="below",
                 border_fill_color="whitesmoke")

    fig.yaxis.axis_label = 'Pixel Row Number'
    fig.xaxis.axis_label = 'Pixel Column Number'

    vlo, lo, hi, vhi = np.nanpercentile(tpf.flux + pedestal, [0.2, 1, 95, 99.8])
    vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    color_mapper = LogColorMapper(palette="Viridis256", low=lo, high=hi)

    fig.image([tpf.flux[fiducial_frame, :, :] + pedestal], x=tpf.column, y=tpf.row,
              dw=tpf.shape[2], dh=tpf.shape[1], dilate=True,
              color_mapper=color_mapper, name="tpfimg")

    # The colorbar will update with the screen stretch slider
    # The colorbar margin increases as the length of the tick labels grows.
    # This colorbar share of the plot window grows, shrinking plot area.
    # This effect is known, some workarounds might work to fix the plot area:
    # https://github.com/bokeh/bokeh/issues/5186
    color_bar = ColorBar(color_mapper=color_mapper,
                         ticker=LogTicker(desired_num_ticks=8),
                         label_standoff=-10, border_line_color=None,
                         location=(0, 0), background_fill_color='whitesmoke',
                         major_label_text_align='left',
                         major_label_text_baseline='middle',
                         title='e/s', margin=0)
    fig.add_layout(color_bar, 'right')

    color_bar.formatter = PrintfTickFormatter(format="%14u")

    if tpf_source is not None:
        fig.rect('xx', 'yy', 1, 1, source=tpf_source, fill_color='gray',
                fill_alpha=0.4, line_color='white')

    # Configure the stretch slider and its callback function
    stretch_slider = RangeSlider(start=np.log10(vlo),
                                 end=np.log10(vhi),
                                 step=vstep,
                                 title='Screen Stretch (log)',
                                 value=(np.log10(lo), np.log10(hi)),
                                 orientation='horizontal',
                                 width=200,
                                 direction='ltr',
                                 show_value=True,
                                 sizing_mode='fixed',
                                 height=15,
                                 name='tpfstretch')

    def stretch_change_callback(attr, old, new):
        """TPF stretch slider callback."""
        fig.select('tpfimg')[0].glyph.color_mapper.high = 10**new[1]
        fig.select('tpfimg')[0].glyph.color_mapper.low = 10**new[0]

    stretch_slider.on_change('value', stretch_change_callback)

    return fig, stretch_slider


