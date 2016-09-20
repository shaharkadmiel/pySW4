# -*- coding: utf-8 -*-
"""
Routines for reading a line or an array of synthetic waveforms.

.. module:: waveforms

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

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
import obspy
from ..headers import STF, REC_MODE

COORDINATE_SYSTEM = {'cartesian'    : ('x','y'),
                     'geographical' : ('lon', 'lat')}


def read_stations(name, path='./', mode='velocity', verbose=False):
    """
    Read a line or an array of stations constructed using
    :func:`~pySW4.prep.stations.station_line` or
    :func:`~pySW4.prep.stations.station_array` in the preprocess phase.

    Parameters
    ----------
    name : str
        The prefix of the line or array name, should be equal to the
        value used for ``'name'`` in
        :func:`~pySW4.prep.stations.station_line` or
        :func:`~pySW4.prep.stations.station_array` in the preprocess
        phase.

    path : str
        Path (relative or absolute) to the files (default is './').

    mode : str
        Mode of the ``rec`` files to be read. One of:

        'displacement', 'velocity', 'div', 'curl', or 'strains'.

    Returns
    -------
    :class:~`Strations`
        Class with waveform data and methods.
    """
    return Stations(name, path, mode, verbose)


class Stations():
    """
    Class to handle waveform data from a line or an array of
    seismograms.

    Read a line or an array of stations constructed using
    :func:`~pySW4.prep.stations.station_line` or
    :func:`~pySW4.prep.stations.station_array` in the preprocess phase.

    Parameters
    ----------
    name : str
        The prefix of the line or array name, should be equal to the
        value used for ``'name'`` in
        :func:`~pySW4.prep.stations.station_line` or
        :func:`~pySW4.prep.stations.station_array` in the preprocess
        phase.

    path : str
        Path (relative or absolute) to the files (default is './').

    mode : str
        Mode of the ``rec`` files to be read. One of:

        'displacement', 'velocity', 'div', 'curl', or 'strains'.
    """
    def __init__(self, name, path='./', mode='velocity', verbose=False):
        self.name = name
        self.mode = mode
        self.traces = None
        self.coordinate_system = None
        self.time = None

        files = glob(os.path.join(path, name) + '*' + REC_MODE[mode])
        if len(files) < 1:
            msg = 'No files were found in {} with pattern {}*{}...'
            raise IOError(msg.format(path, name, REC_MODE[mode]))
        self.traces = obspy.Stream()
        xi, yi = [], []
        for f in files:
            if verbose:
                print('Processing {}'.format(f))

            # parse the filename to get the location of the trace
            xcoor, x, ycoor, y, zcoor, z = self._parse_rec_filename(f)
            trace = obspy.read(f, format='SAC')[0]

            # add trace location to the trace.stats object
            trace.stats[xcoor] = x
            trace.stats[ycoor] = y
            trace.stats[zcoor] = z

            self.traces += trace

        if xcoor in (COORDINATE_SYSTEM['cartesian']):
            self.coordinate_system = 'cartesian'
        elif xcoor in (COORDINATE_SYSTEM['geographical']):
            self.coordinate_system = 'geographical'

        self.time = trace.times()

    def _parse_rec_filename(self, filename):
        """
        Parse 'rec' filename and get the x, y or lat, lon coordinates.
        """
        x, y, z = filename.rsplit('_', 4)[1:4]
        xcoor, x = x.split('=')
        ycoor, y = y.split('=')
        zcoor, z = z.split('=')
        return xcoor, float(x), ycoor, float(y), zcoor, float(z)
