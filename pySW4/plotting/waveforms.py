# -*- coding: utf-8 -*-
"""
Plotting routines for observed and synthetic waveforms.

.. module:: waveforms

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


.. rubric:: SW4 uses a right handed coordinate system


::

         X
       ⋰
     ⋰
    o------->Y
    |
    |
    V
    Z

If the azimuth of the SW4 grid is ``0`` this translates to:

- X == Northing
- Y == Easting
- Z == Vertical (inverted!)
"""
from __future__ import absolute_import, print_function, division

import os
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.gridspec import GridSpec
import numpy as np
import obspy
from ..postp.input import _parse_input_file_and_folder
from ..headers import STF
from ..utils import fourier_spectrum

MODE = {'displacement' : 'displacement, m' ,
        'velocity'     : 'velocity, m/s'   ,
        'div'          : 'div'             ,
        'curl'         : 'curl'            ,
        'strains'      : 'strain'          }


def plot_traces(traces, mode='', yscale='auto', hspace=0.2, wspace=0.05,
                figsize=None, fig=None, color='k'):
    """
    Plot all traces and their Fourier specra side-by-side.

    Parameters
    ----------
    traces : :class:`~obspy.Stream`
        Traces to be plotted in an :class:`~obspy.Stream` object.

    mode : str
        Mode describes the type of data in traces. One of:

        'displacement', 'velocity', 'div', 'curl', or 'strains'.

        Optionaly, an alternative string can be given that will be used
        as the y-label of the time-histories.

    yscale : str
        Set the scale of the vertical y-axis:

        - ``auto`` - Vertical scale of each axes is automatically set to
        the -|max| and |max| of each trace.
        - ``all`` - Vertical scale of all axes is set to the same limits
        which are the -|max| and |max| of all traces.
        - ``normalize`` - Each trace is normalized and plotted
        (``ylim=(-1, 1)``).

    hspace : float
        The hight space between axes. See
        :class:`~matplotlib.gridspec.GridSpec` documentation.

    wspace : float
        The width space between axes. See
        :class:`~matplotlib.gridspec.GridSpec` documentation.

    figsize : 2-tuple
        Size of the :class:`~matplotlib.figure.Figure`.

    fig : :class:`~matplotlib.figure.Figure`
        A :class:`~matplotlib.figure.Figure` instance.

    color : str
        Color of the line in the time-histories and Fourier spectra.
    """
    count = traces.count()

    if not fig:
        set_title = True
        if not figsize:
            figsize = (6, 2 * int(count))

        gs = GridSpec(count, 2, width_ratios=[2, 1],
                      hspace=hspace, wspace=wspace)
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(item) for item in gs]
    else:
        ax = fig.get_axes()
        set_title = False

    _min, _max = -max(np.abs(traces.max())), max(np.abs(traces.max()))
    for i, tr in enumerate(traces.copy()):
        # plot seismogram
        axi = ax[i * 2]
        if yscale is 'auto':
            axi.set_ylim(-np.abs(tr.max()),
                         np.abs(tr.max()))
        elif yscale is 'all':
            axi.set_ylim(_min, _max)
        elif yscale is 'normalize':
            tr.normalize()
            axi.set_ylim(-1, 1)
        axi.yaxis.tick_left()
        if set_title:
            axi.text(0.99, 0.97,
                     tr.stats.channel,
                     transform=axi.transAxes, ha='right', va='top',
                     fontsize=10)

        try:
            axi.set_ylabel(MODE[mode])
        except KeyError:
            axi.set_ylabel(mode)

        axi.plot(tr.times(), tr.data, color=color)
        axi.set_xlim(tr.times()[0], tr.times()[-1])

        # plot spectrum
        axi = ax[i * 2 + 1]
        freq, amp = fourier_spectrum(tr)
        axi.loglog(freq, amp, color=color)
        axi.yaxis.tick_right()
        axi.grid(True)
        axi.set_xlim(freq[0], freq[-1])

    for axi in ax[:-2]:
        axi.set_xticklabels([])

    axi = ax[-2]
    axi.set_xlabel('time, s')
    axi = ax[-1]
    axi.set_xlabel('frequency, Hz')

    return fig, ax


def create_seismogram_plots(
        input_file, folder=None, stream_observed=None, inventory=None,
        water_level=None, pre_filt=None, filter_kwargs=None,
        channel_map={"-Vz": "Z", "Vx": "N", "Vy": "E"}, used_stations=None,
        synthetic_starttime=None):
    """
    Create all waveform plots, comparing synthetic and observed data.

    Ideally works on a SW4 input file, or explicitely on an folder with
    SW4 output files. Assumes output in SAC format. Observed/real data
    and station metadata can be specified, along with deconvolution
    parameters to get to physical units.

    Parameters
    ----------
    input_file : str
        Filename (potentially with absolute/relative path) of SW4 input
        file used to control the simulation. Use ``None`` to work on
        folder with SW4 output without using metadata from input file.

    folder : str
        Folder with SW4 output files or ``None`` if output folder
        location can be used from input file. Only needed when no input
        file is specified or if output folder was moved to a different
        location after the simulation.

    stream_observed : :class:`obspy.core.stream.Stream`
        Observed/real data to compare with synthetics.

    inventory : :class:`obspy.core.inventory.inventory.Inventory`
        Station metadata for observed/real data.

    water_level : float
        Water level for instrument response removal (see
        :meth:`obspy.core.trace.remove_response`).

    pre_filt : 4-tuple of float
        Frequency domain pre-filtering in response removal (see
        :meth:`obspy.core.trace.Trace.remove_response`).

    filter_kwargs : dict
        Filter parameters for filtering applied to observed data after
        response removal before comparison to synthetic data. ``kwargs``
        are passed on to :meth:`obspy.core.stream.Stream.filter`.

    channel_map : dict
        Mapping dictionary to match synthetic channel to component in
        observed data.

    used_stations : list
        Station codes to consider in plot output. Use all stations if
        left ``None``.

    synthetic_starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
        Start time of synthetic data, only needed if no input file is
        specified or if input file did not set the correct origin time
        of the event.
    """
    input, folder = _parse_input_file_and_folder(input_file, folder)

    stf_type = STF[input.source[0].type].type
    if stf_type == 'displacement':
        evalresp_output = "DISP"
        unit_label = "m"
    elif stf_type == 'velocity':
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


def _plot_seismograms(st_synth_, st_real_, channel_map, unit_label,
                      outfile, figsize=None):
    """
    Helper function that plots synthetic vs. real data to an image file.

    Parameters
    ----------
    st_synth : :class:`obspy.core.stream.Stream`
        Synthetic waveform data.

    st_real : :class:`obspy.core.stream.Stream`
        Observed waveform data.

    channel_map : dict
        Mapping dictionary to match synthetic channel to component in
        observed data.

    unit_label : str
        Label string for y-axis of waveforms.

    outfile : str
        Output filename (absolute or relative path) for image including
        suffix (e.g. png).

    figsize : 2-tuple of floats
        Matplotlib figure size (inches x/y).
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
