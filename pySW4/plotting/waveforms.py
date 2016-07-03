# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import os
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import obspy
from ..core.config import _parse_config_file_and_folder
from ..core.header import SOURCE_TIME_FUNCTION_TYPE


# SW4 uses a right handed coordinate system:
#   - X == Northing
#   - Y == Easting
#   - Z == Vertical (inverted!)
#
#          X
#        ⋰
#      ⋰
#     o------->Y
#     |
#     |
#     V
#     Z
def create_seismogram_plots(
        config_file, folder=None, stream_observed=None, inventory=None,
        water_level=None, pre_filt=None, filter_kwargs=None,
        channel_map={"-Vz": "Z", "Vx": "N", "Vy": "E"}, used_stations=None,
        synthetic_starttime=None):
    """
    Create all waveform plots, comparing synthetic and observed data.

    Ideally works on a SW4 input/config file, or explicitely on an folder with
    SW4 output files. Assumes output in SAC format. Observed/real data and
    station metadata can be specified, along with deconvolution parameters to
    get to physical units.

    :type config_file: str
    :param config_file: Filename (potentially with absolute/relative path) of
        SW4 input/config file used to control the simulation. Use `None` to
        work on folder with SW4 output without using metadata from config.
    :type folder: str
    :param folder: Folder with SW4 output files or `None` if output folder
        location can be used from config file. Only needed when no config file
        is specified or if output folder was moved to a different location
        after the simulation.
    :type stream_observed: :class:`obspy.core.stream.Stream`
    :param stream_observed: Observed/real data to compare with synthetics.
    :type inventory: :class:`obspy.core.inventory.inventory.Inventory`
    :param inventory: Station metadata for observed/real data.
    :type water_level: float
    :param water_level: Water level for instrument response removal (see
        :meth:`obspy.core.trace.remove_response`).
    :type pre_filt: 4-tuple of float
    :param pre_filt: Frequency domain pre-filtering in response removal (see
        :meth:`obspy.core.trace.Trace.remove_response`).
    :type filter_kwargs: dict
    :param filter_kwargs: Filter parameters for filtering applied to observed
        data after response removal before comparison to synthetic data. Kwargs
        are passed on to :meth:`obspy.core.stream.Stream.filter`).
    :type channel_map: dict
    :param channel_map: Mapping dictionary to match synthetic channel to
        component in observed data.
    :type used_stations: list
    :param used_stations: Station codes to consider in plot output. Use all
        stations if left `None`.
    :type synthetic_starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param synthetic_starttime: Start time of synthetic data, only needed if no
        config file is specified or if config file did not set the correct
        origin time of the event.
    """
    config, folder = _parse_config_file_and_folder(config_file, folder)

    stf_type = SOURCE_TIME_FUNCTION_TYPE[config.source[0].type]
    if stf_type == 0:
        evalresp_output = "DISP"
        unit_label = "m"
    elif stf_type == 1:
        evalresp_output = "VEL"
        unit_label = "m/s"
    else:
        raise NotImplementedError()

    st_synth = obspy.read(os.path.join(folder, "*.?v"))
    st_real = stream_observed or obspy.Stream()
    if used_stations is not None:
        st_synth.traces = [tr for tr in st_synth
                           if tr.stats.station in used_stations]
    stations = set([tr.stats.station for tr in st_synth])
    st_real.traces = [tr for tr in st_real
                      if tr.stats.station in stations]
    for tr in st_synth:
        # SW4 vertical channel is positive in coordinate direction, which
        # is depth positive down. So we have to invert it to get the normal
        # seismometer vertical up trace.
        if tr.stats.channel == "Vz":
            tr.stats.channel = "-Vz"
            tr.data *= -1
    t_min = min([tr.stats.starttime for tr in st_synth])
    t_max = min([tr.stats.endtime for tr in st_synth])
    st_real.attach_response(inventory)
    st_real.remove_response(
        output=evalresp_output, water_level=water_level, pre_filt=pre_filt)
    if filter_kwargs:
        st_real.filter(**filter_kwargs)
    st_real.trim(t_min, t_max)

    outfile = os.path.join(folder, "seismograms.png")
    _plot_seismograms(st_synth, st_real, channel_map, unit_label, outfile)
    for station in stations:
        outfile = os.path.join(folder, "seismograms.{}.png".format(station))
        st_synth_ = st_synth.select(station=station)
        st_real_ = st_real.select(station=station)
        _plot_seismograms(
            st_synth_, st_real_, channel_map, unit_label, outfile,
            figsize=(10, 8))


def _plot_seismograms(
        st_synth_, st_real_, channel_map, unit_label, outfile, figsize=None):
    """
    Helper function that plots synthetic vs. real data to an image file.

    :type st_synth: :class:`obspy.core.stream.Stream`
    :param st_synth: Synthetic waveform data.
    :type st_real: :class:`obspy.core.stream.Stream`
    :param st_real: Observed waveform data.
    :type channel_map: dict
    :param channel_map: Mapping dictionary to match synthetic channel to
        component in observed data.
    :type unit_label: str
    :param unit_label: Label string for y-axis of waveforms.
    :type outfile: str
    :param outfile: Output filename (absolute or relative path) for image
        including suffix (e.g. png).
    :type figsize: 2-tuple of floats
    :param figsize: Matplotlib figure size (inches x/y).
    """
    if figsize is None:
        figsize = (10, len(st_synth_))

    fig = plt.figure(figsize=figsize)
    st_synth_.plot(fig=fig)

    for ax in fig.axes:
        id = ax.texts[0].get_text()
        _, sta, _, cha = id.split(".")
        real_component = channel_map[cha]
        # find appropriate synthetic trace
        for tr_real in st_real_:
            if tr_real.stats.station != sta:
                continue
            # SW4 synthetics channel codes (for velocity traces) are "V[xyz]"
            if tr_real.stats.channel[-1] != real_component:
                continue
            break
        else:
            continue
        ax.text(0.95, 0.9, tr_real.id, ha="right", va="top", color="r",
                transform=ax.transAxes)
        t = date2num([tr_real.stats.starttime + t_ for
                      t_ in tr_real.times()])
        ax.plot(t, tr_real.data, "r-")
        ax.set_ylabel(unit_label)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, hspace=0.0, wspace=0.0)
    fig.savefig(outfile)
    plt.close(fig)
