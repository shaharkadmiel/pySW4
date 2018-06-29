# -*- coding: utf-8 -*-
"""
Python module for placing stations to record synthetic seismograms in
SW4 simulations.

.. module:: stations

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import os
import shutil
import numpy as np
from obspy import read_inventory, Inventory
from warnings import warn


def _append2infile(infile, name, string):
    """
    Helper function. Copy an existing SW4 input file, add `name`
    to the filename, open the new file in append mode and write the
    string to the end of the file.
    """
    if name.join(('.', '.')) not in infile:
        filename, extention = os.path.splitext(infile)
        filename += '.' + name + extention
        shutil.copyfile(infile, filename)
    else:
        filename = infile

    with open(filename, "a") as f:
        f.write(string)
    f.close()

    return filename


def station_array(x1=None, x2=None, y1=None, y2=None,
                  lon1=None, lon2=None, lat1=None, lat2=None,
                  depth=0, number_of_stations=None, spacing=None,
                  name='array', mode='displacement', writeEvery=100,
                  nsew=0, fmt='%.3f', infile=None):

    """
    Place stations to record synthetics on a line.

    Corners of the array can be given in -

    **Catersian grid coordinates:**

    `x1`, `y1`, `x2`, `y2`

    or in -

    **Geographical coordinates:**

    `lon1`, `lat1`, `lon2`, `lat2`

    Parameters
    ----------
    depth : float
        Depth of the stations in the array.

    number_of_stations : int or tuple or None
        If `int`, number of stations is the same in both directions. To
        set a different number of stations in each direction provide a
        `tuple`. If ``None``, the `spacing` argument is expected.

    spacing : float or tuple or None
        If `float`, spacing between stations is taken to be the same in
        both directions. To set different spacing in each direction
        provide a `tuple`. If ``None``, the `number_of_stations`
        argument is expected.

        `spacing` must be given in meters if Cartesian grid
        coordinates are used or in decimal degrees if Geographical
        coordinates are used.

    name : str
        Prefix name of the sac file name.

    mode : {'displacement' (default), 'velocity', 'div', 'curl', 'strains'}
        Mode of the recorded motions.

    writeEvery : int
        Cycle interval to write the data to disk.

    nsew : int
        The components of the station:

        - 0 - x, y, z (default), or
        - 1 - East, North, Vertical

    fmt : str
        Format of the coordinates, default is '%.3f'.

    infile : str
        Path (relative or absolute) to the SW4 inputfile. Stations are
        added to a copy of the specified file and saved with the string
        `name` appended to the `infile` name.
        (`name` is 'array' by default).

        If set to ``None``, a formated string is returned that can then
        be copied manually to an input file.

    Returns
    -------
    str
        Filename of the newly saved inputfile or a formatted string that
        can be copied manually to an SW4 inputfile.
    """
    if not number_of_stations and not spacing:
        raise ValueError(
            'Either ``number_of_stations`` or ``spacing`` must be specified!'
        )
    elif number_of_stations and spacing:
        warnings.warn(
            'Both ``number_of_stations`` and ``spacing`` are specified, '
            'using ``number_of_stations={}``.'.format(number_of_stations))
    elif number_of_stations:
        try:
            number_of_stations = int(number_of_stations)
            number_of_stations = (number_of_stations, number_of_stations)
        except TypeError:
            pass
    elif spacing:
        try:
            spacing = float(spacing)
            spacing = (spacing, spacing)
        except TypeError:
            pass

    if x1 is not None and x2 is not None:
        try:
            x = np.linspace(x1, x2, number_of_stations[0])
            y = np.linspace(y1, y2, number_of_stations[1])
        except TypeError:
            x = np.arange(x1, x2, spacing[0])
            y = np.arange(y1, y2, spacing[1])

        sac_string = ('rec x={0} y={0} depth={0} '
                      'file=%s_x={0}_y={0}_z={0}_ '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    elif lon1 is not None and lon2 is not None:
        try:
            x = np.linspace(lon1, lon2, number_of_stations[0])
            y = np.linspace(lat1, lat2, number_of_stations[1])
        except TypeError:
            x = np.arange(lon1, lon2, spacing[0])
            y = np.arange(lat1, lat2, spacing[1])

        sac_string = ('rec lon={0} lat={0} depth={0} '
                      'file=%s_lon={0}_lat={0}_z={0}_ '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    points = [(i, j) for j in y for i in x]

    string = ('\n\n# %dx%d array of seismograms added: %s\n'
              % (len(x), len(y), name))

    for (i, j) in points:
        string += (sac_string % (i, j, depth, name, i, j, depth,
                                 writeEvery, nsew, mode))

    if infile is None:
        return string
    else:
        return _append2infile(infile, name, string)


def station_line(x1=None, x2=None, y1=None, y2=None,
                 lon1=None, lon2=None, lat1=None, lat2=None,
                 depth1=0, depth2=0, number_of_stations=3,
                 name='line', mode='displacement', writeEvery=100,
                 nsew=0, fmt='%.3f', infile=None):
    """
    Place stations to record synthetics on a line.

    Start- and end-point can be given in -

    **Catersian grid coordinates:**

    `x1`, `y1`, `depth1`, `x2`, `y2`, `depth2`

    or in -

    **Geographical coordinates:**

    `lon1`, `lat1`, `depth1`, `lon2`, `lat2`, `depth2`

    Parameters
    ----------
    number_of_stations : int
        Number of stations to place on the line (defaults to 3 - one at
        each end and another one in the middle).

    name : str
        Prefix name of the sac file name.

    mode : {'displacement' (default), 'velocity', 'div', 'curl', 'strains'}
        Mode of the recorded motions.

    writeEvery : int
        Cycle interval to write the data to disk.

    nsew : int
        The components of the station:

        - 0 - x, y, z (default), or
        - 1 - East, North, Vertical

    fmt : str
        Format of the coordinates, default is '%.3f'.

    infile : str
        Path (relative or absolute) to the SW4 inputfile. Stations are
        added to a copy of the specified file and saved with the string
        `name` appended to the `infile` name.
        (`name` is 'array' by default).

        If set to ``None``, a formated string is returned that can then
        be copied manually to an input file.

    Returns
    -------
    str
        Filename of the newly saved inputfile or a formatted string that
        can be copied manually to an SW4 inputfile.
    """

    if x1 is not None and x2 is not None:
        x = np.linspace(x1, x2, number_of_stations)
        y = np.linspace(y1, y2, number_of_stations)

        sac_string = ('rec x={0} y={0} depth={0} '
                      'file=%s_x={0}_y={0}_z={0}_ '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    elif lon1 is not None and lon2 is not None:
        x = np.linspace(lon1, lon2, number_of_stations)
        y = np.linspace(lat1, lat2, number_of_stations)

        sac_string = ('rec lon={0} lat={0} depth={0} '
                      'file=%s_lon={0}_lat={0}_z={0}_ '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    z = np.linspace(depth1, depth2, number_of_stations)

    string = '\n\n# stations on a line: %s\n' % name
    for i in range(len(x)):
        string += (sac_string % (x[i], y[i], z[i], name,
                                 x[i], y[i], z[i], writeEvery, nsew, mode))

    if infile is None:
        return string
    else:
        return _append2infile(infile, name, string)


def station_location(x=None, y=None, lat=None, lon=None, depth=0,
                     name='st', mode='displacement', writeEvery=100,
                     nsew=0, fmt='%.3f', infile=None):
    """
    Place stations to record synthetics at specific locations.

    Can handle a single station or a list of stations. If several
    stations are passes, `x, y` (or `lat, lon`), and `name` must
    be the same length.

    Locations may be given in -

    **Catersian grid coordinates:**

    `x`, `y`, `depth`

    or in -

    **Geographical coordinates:**

    `lon`, `lat`, `depth`

    Parameters
    ----------
    name : str
        Prefix name of the sac file name.

    mode : {'displacement' (default), 'velocity', 'div', 'curl', 'strains'}
        Mode of the recorded motions.

    writeEvery : int
        Cycle interval to write the data to disk.

    nsew : int
        The components of the station:

        - 0 - x, y, z (default), or
        - 1 - East, North, Vertical

    fmt : str
        Format of the coordinates, default is '%.3f'.

    infile : str
        Path (relative or absolute) to the SW4 inputfile. Stations are
        added to a copy of the specified file and saved with the string
        `name` appended to the `infile` name.
        (`name` is 'array' by default).

        If set to ``None``, a formated string is returned that can then
        be copied manually to an input file.

    Returns
    -------
    str
        Filename of the newly saved inputfile or a formatted string that
        can be copied manually to an SW4 inputfile.
    """

    if x is not None and y is not None:
        sac_string = ('rec x={0} y={0} depth={0} file=%s '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)
    elif lon is not None and lat is not None:
        x, y = lon, lat
        sac_string = ('rec lon={0} lat={0} depth={0} file=%s '
                      'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    string = '\n\n# stations at locations:\n'
    try:
        for i in range(len(x)):
            string += (sac_string % (float(x[i]), float(y[i]), depth,
                                     name[i], writeEvery, nsew, mode))
    except TypeError:
        string += (sac_string % (float(x), float(y), depth, name,
                                 writeEvery, nsew, mode))
    if infile is None:
        return string
    else:
        return _append2infile(infile, name, string)


def inventory2station_locations(inv, mode='displacement', writeEvery=100,
                                name='st', nsew=0, fmt='%.3f', infile=None):
    """
    Place stations to record synthetics at specific locations.

    Extracts station locations from an
    :class:`~obspy.core.inventory.inventory.Inventory` object and
    generate an SW4 inputfile string.

    Parameters
    ----------

    inv : str or :class:`~obspy.core.inventory.inventory.Inventory`
        Path to a StationXML file or an
        :class:`~obspy.core.inventory.inventory.Inventory` object.

    mode : {'displacement' (default), 'velocity', 'div', 'curl', 'strains'}
        Mode of the recorded motions.

    writeEvery : int
        Cycle interval to write the data to disk.

    nsew : int
        The components of the station:

        - 0 - x, y, z (default), or
        - 1 - East, North, Vertical

    fmt : str
        Format of the coordinates, default is '%.3f'.

    infile : str
        Path (relative or absolute) to the SW4 inputfile. Stations are
        added to a copy of the specified file and saved with the string
        `name` appended to the `infile` name.
        (`name` is 'array' by default).

        If set to ``None``, a formated string is returned that can then
        be copied manually to an input file.

    Returns
    -------
    str
        Filename of the newly saved inputfile or a formatted string that
        can be copied manually to an SW4 inputfile.
    """

    if not isinstance(inv, Inventory):
        inv = read_inventory(inv)

    sac_string = ('rec lon={0} lat={0} depth=0 file=%s '
                  'writeEvery=%d nsew=%d variables=%s\n').format(fmt)

    string = '\n\n# stations at locations:\n'
    for net in inv.networks:
        for sta in net.stations:
            string += (sac_string % (sta.longitude, sta.latitude,
                                     '.'.join((net.code, sta.code)),
                                     writeEvery, nsew, mode))
    if infile is None:
        return string
    else:
        return _append2infile(infile, name, string)
