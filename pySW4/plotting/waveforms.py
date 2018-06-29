# -*- coding: utf-8 -*-
"""
Plotting routines for observed and synthetic waveforms.

.. module:: waveforms

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

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

import glob
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.gridspec import GridSpec
import numpy as np
import obspy
from ..sw4_metadata import _parse_input_file_and_folder
from ..headers import STF
from ..utils import fourier_spectrum

MODE = {'displacement' : 'displacement, m' ,
        'velocity'     : 'velocity, m/s'   ,
        'div'          : 'div'             ,
        'curl'         : 'curl'            ,
        'strains'      : 'strain'          }


def plot_traces(traces, mode='', yscale='auto', hspace=0.2, wspace=0.05,
                figsize=None, fig=None, **kwargs):
    """
    Plot all traces and their Fourier specra side-by-side.

    Parameters
    ----------
    traces : :class:`~obspy.Stream`
        Traces to be plotted in an :class:`~obspy.Stream` object.

    mode : {'displacement', 'velocity', 'div', 'curl', 'strains'}
        Mode describes the type of data in traces.

        Optionaly, an alternative string can be given that will be used
        as the y-label of the time-histories.

    yscale : {'auto', 'all', 'normalize'}
        Set the scale of the vertical y-axis:

        - ``auto`` - Vertical scale of each axes is automatically set to
          the -\|max\| and \|max\| of each trace.

        - ``all`` - Vertical scale of all axes is set to the same limits
          which are the -\|max\| and \|max\| of all traces.

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

    Other Parameters
    ----------------
    kwargs : :func:`~matplotlib.pyplot.plot` propeties.
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
            location = ''
            for k, v in zip(tr.stats.coordinates.keys(),
                            tr.stats.coordinates.values()):
                location += '{}\n'.format(k + '=' + str(v))
            axi.text(0.99, 0.97,
                     '{}:\n{}'.format(tr.id.split('..')[0], location),
                     color='r',
                     transform=axi.transAxes, ha='right', va='top',
                     fontsize=8)
            axi.text(0.01, 0.03,
                     tr.stats.starttime, color='r',
                     transform=axi.transAxes, ha='left', va='bottom',
                     fontsize=8)

        try:
            axi.set_ylabel(MODE[mode])
        except KeyError:
            axi.set_ylabel(mode)

        axi.plot(tr.times(), tr.data, **kwargs)
        axi.set_xlim(tr.times()[0], tr.times()[-1])

        # plot spectrum
        axi = ax[i * 2 + 1]
        freq, amp = fourier_spectrum(tr)
        axi.loglog(freq, amp, **kwargs)
        axi.yaxis.tick_right()
        axi.grid(True, ls='dashed')
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
        channel_map=None, used_stations=None, synthetic_starttime=None,
        synthetic_data_glob='*.?v', t0_correction_fraction=0.0,
        synthetic_scaling=False, verbose=False):
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

    synthetic_starttime : :class:`obspy.core.utcdatetime.UTCDateTime`
        Start time of synthetic data, only needed if no input file is
        specified or if input file did not set the correct origin time
        of the event.

    synthetic_data_glob : str
        Glob pattern to lookup synthetic data. Use e.g.
        '\*.[xyz]' or '\*.?' for synthetic data saved as "displacement"
        (the solution of the forward solver), or '\*.?v' for synthetic
        data saved as "velocity" (the differentiated solution of the
        forward solver).

    t0_correction_fraction : float
        Fraction of t0 used in SW4 simulation
        (offset of source time function to prevent solver artifacts) to
        account for (i.e. shift synthetics left to earlier absolute
        time). '0.0' means no correction of synthetics time is done,
        '1.0' means that synthetic trace is shifted left in time by
        ``t0`` of SW4 run.

    synthetic_scaling : bool or float
        Scaling to apply to synthetic seismograms. If
        ``False``, no scaling is applied. If a float is provided, all
        synthetics' will be scaled with the given factor (e.g. using
        ``2.0`` will scale up synthetics by a factor of 2).
    """
    input_, folder = _parse_input_file_and_folder(input_file, folder)

    stf_type = STF[input_.source[0].type].type
    if stf_type == 'displacement':
        evalresp_output = "DISP"
        unit_label = "m"
    elif stf_type == 'velocity':
        evalresp_output = "VEL"
        unit_label = "m/s"
    else:
        raise NotImplementedError()

    if verbose:
        print('Correcting observed seismograms to output "{}" using '
              'evalresp.'.format(evalresp_output))

    if channel_map is None:
        channel_map = {
            "-Vz": "Z", "Vx": "N", "Vy": "E",
            "-Z": "Z", "X": "N", "Y": "E"}

    info_text = ''

    files_synth = glob.glob(os.path.join(folder, synthetic_data_glob))
    st_synth = obspy.Stream()
    for file_ in files_synth:
        st_synth += obspy.read(file_)
    if t0_correction_fraction:
        if len(config.source) > 1:
            msg = ('t0_correction_fraction is not implemented for SW4 run '
                   'with multiple sources.')
            raise NotImplementedError(msg)
        t0 = config.source[0].t0
        t0_correction = t0 * t0_correction_fraction
        info_text += (
            ' Synthetics start time corrected by {}*t0 '
            '(-{}s).').format(t0_correction_fraction, t0_correction)
        for tr in st_synth:
            tr.stats.starttime -= t0_correction
    if synthetic_scaling is not False:
        info_text += (' Synthetics scaled with a factor of {}.').format(
            synthetic_scaling)
        for tr in st_synth:
            tr.data *= synthetic_scaling
        if verbose:
            print('Scaling synthetics by {}'.format(synthetic_scaling))

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
        elif tr.stats.channel == "Z":
            tr.stats.channel = "-Z"
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
    _plot_seismograms(st_synth, st_real, channel_map, unit_label, outfile,
                      info_text=info_text)
    for station in stations:
        outfile = os.path.join(folder, "seismograms.{}.png".format(station))
        st_synth_ = st_synth.select(station=station)
        st_real_ = st_real.select(station=station)
        _plot_seismograms(
            st_synth_, st_real_, channel_map, unit_label, outfile,
            figsize=(10, 8), info_text=info_text)


def _plot_seismograms(
        st_synth_, st_real_, channel_map, unit_label, outfile, figsize=None,
        info_text=None):
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
        real_component = channel_map.get(cha)
        if real_component is None:
            real_component = cha[-1]
            msg = ('Synthetic channel could not be mapped to component code '
                   'of observed data. Using last character of synthetic data '
                   '("{}") to match observed data component code. Are data '
                   'already rotated by SW4 through config '
                   'file settings?').format(real_component)
            warnings.warn(msg)
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
        t = date2num([(tr_real.stats.starttime + t_).datetime
                      for t_ in tr_real.times()])
        ax.plot(t, tr_real.data, "r-")
        ax.set_ylabel(unit_label)
    if info_text:
        ax.text(0.95, 0.02, info_text, ha="right", va="bottom", color="b",
                transform=ax.transAxes)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, hspace=0.0, wspace=0.0)
    fig.savefig(outfile)
    plt.close(fig)
