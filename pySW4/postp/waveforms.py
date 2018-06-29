# -*- coding: utf-8 -*-
"""
Routines for reading a line or an array of synthetic waveforms.

.. module:: waveforms

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

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
from glob import glob
import numpy as np
from obspy import read, Stream
from obspy.core.util import AttribDict
from ..headers import STF, REC_MODE
from ..plotting import plot_traces

COORDINATE_SYSTEM = {'cartesian'    : ('x','y'),
                     'geographical' : ('lon', 'lat')}


def read_stations(name, path='./', mode='velocity', verbose=False):
    """
    Read a single station or several stations along a line or as an
    array  constructed using :func:`~pySW4.prep.stations.station_line`
    or :func:`~pySW4.prep.stations.station_array` in the preprocess
    phase.

    Parameters
    ----------
    name : str
        For a single station give the name of the station used in the
        inputfile in the ``file`` field. For all stations relating to a
        line or array of stations use the prefix of the line or array
        name, should be equal to the value used for ``'name'`` in
        :func:`~pySW4.prep.stations.station_line` or
        :func:`~pySW4.prep.stations.station_array` in the preprocess
        phase.

    path : str
        Path (relative or absolute) to the files (default is './').

    mode : {'displacement', 'velocity', 'div', 'curl', 'strains'}
        Mode of the ``rec`` files to be read.

    Returns
    -------
    :class:`~.Stations`
        Class with waveform data and methods.
    """
    return Stations(name, path, mode, verbose)


class Stations(Stream):
    """
    Class to handle waveform data from a line or an array of
    seismograms.

    Read a line or an array of stations constructed using
    :func:`~pySW4.prep.stations.station_line` or
    :func:`~pySW4.prep.stations.station_array` in the preprocess phase.

    Parameters
    ----------
    name : str
        For a single station give the name of the station used in the
        inputfile in the ``file`` field. For all stations relating to a
        line or array of stations use the prefix of the line or array
        name, should be equal to the value used for ``'name'`` in
        :func:`~pySW4.prep.stations.station_line` or
        :func:`~pySW4.prep.stations.station_array` in the preprocess
        phase.

    path : str
        Path (relative or absolute) to the files (default is './').

    mode : {'displacement', 'velocity', 'div', 'curl', 'strains'}
        Mode of the ``rec`` files to be read.
    """
    def __init__(self, name=None, path='./', mode='velocity', verbose=False,
                 traces=None):
        super(Stations, self).__init__(traces)

        try:
            self.name = name
            self.mode = mode

            files = glob(os.path.join(path, name) + '*' + REC_MODE[mode])
            if len(files) < 1:
                msg = 'No files were found in {} with pattern {}*{}...'
                raise IOError(msg.format(path, name, REC_MODE[mode]))
            self.xi = []
            self.yi = []
            for i, f in enumerate(files):
                if verbose:
                    print('Processing {}'.format(f))

                # parse the filename to get the location of the trace
                try:
                    (network, station,
                     xcoor, x, ycoor, y, zcoor,
                     z) = self._parse_rec_filename(f)
                except ValueError:
                    try:
                        network = f.rsplit('/', 1)[-1]
                        network, station = network.rsplit('.', 1)
                        xcoor = 'longitude'
                        ycoor = 'latitude'
                        zcoor = 'depth'
                    except ValueError:
                        station = None

                trace = read(f, format='SAC')[0]

                # add coordinates to the xi, yi vectors
                try:
                    if x not in self.xi:
                        self.xi += [x]
                    if y not in self.yi:
                        self.yi += [y]
                except NameError:
                    x = trace.stats.sac.stlo
                    y = trace.stats.sac.stla
                    z = None

                # add coordinates AttribDict to the trace.stats object
                trace.stats.coordinates = AttribDict({
                    xcoor: x,
                    ycoor: y,
                    zcoor: z})

                # update network and station names in trace stats
                trace.stats.network = network
                if station is None:
                    station = str(i)
                trace.stats.station = station

                self.append(trace)

            if xcoor in (COORDINATE_SYSTEM['cartesian']):
                self.coordinate_system = 'cartesian'
            elif xcoor in (COORDINATE_SYSTEM['geographical']):
                self.coordinate_system = 'geographical'

            self.xi.sort()
            self.yi.sort()
            try:
                dx = np.gradient(self.xi)[0]
                dy = np.gradient(self.yi)[0]
            except ValueError:
                dx = 0
                dy = 0
            self.extent = (self.yi[0] - 0.5 * dy, self.yi[-1] + 0.5 * dy,
                           self.xi[0] - 0.5 * dx, self.xi[-1] + 0.5 * dx)

        except AttributeError:
            pass

    def plot_traces(self, mode='', yscale='auto', hspace=0.2,
                    wspace=0.05, figsize=None, fig=None, **kwargs):
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
        return plot_traces(self, mode, yscale, hspace, wspace, figsize, fig,
                           **kwargs)

    def times(self):
        return self.traces[0].times()

    @property
    def starttime(self):
        return self.traces[0].stats.starttime

    @property
    def delta(self):
        return self.traces[0].stats.delta

    def get_data(self, channel='*z'):
        """
        Get data for the selected `channel`. Wildcard are also
        handled such that ``channel='*z'`` gets all vertical channels
        in *self.traces*.

        See :meth:`~obspy.core.stream.Stream.select` method for more
        info.

        Returns
        -------
        3D :class:`~numpy.ndarray`
            3darray of shape (nx, ny, time).

        """
        nx = len(self.xi)
        ny = len(self.yi)
        data = np.empty((nx, ny, self.times().size))
        traces = self.select(channel=channel)

        xcoor = COORDINATE_SYSTEM[self.coordinate_system][0]
        ycoor = COORDINATE_SYSTEM[self.coordinate_system][1]

        for trace in traces:
            data[self.xi.index(trace.stats.coordinates[xcoor]),
                 self.yi.index(trace.stats.coordinates[ycoor])] = trace.data
        return data

    def _parse_rec_filename(self, filename):
        """
        Private method: Parse 'rec' filename and get the x, y or
        lat, lon coordinates.
        """
        network, x, y, z = filename.rsplit('_', 4)[0:4]
        network = network.rsplit('/', 1)[-1]
        try:
            network, station = network.rsplit('.', 1)
        except ValueError:
            station = None

        xcoor, x = x.split('=')
        ycoor, y = y.split('=')
        zcoor, z = z.split('=')

        return (network, station,
                xcoor, float(x), ycoor, float(y), zcoor, float(z))
