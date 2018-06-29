# -*- coding: utf-8 -*-
"""
Parsing routines for SW4 input and output files and directories.

.. module:: sw4_metadata

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
"""
from __future__ import absolute_import, print_function, division

import os
from warnings import warn
from obspy.core.util import AttribDict
import numpy as np
from .utils import nearest_values, geographic2cartesian


class Inputfile(AttribDict):
    """
    A container for the simulation metadata parsed from the SW4
    inputfile.

    Parameters
    ----------
    filename : str or :class:`~obspy.core.util.attribdict.AttribDict`
        Path (relative or absolute) of an SW4 input file.
    """
    def __init__(self, filename):
        if type(filename) is str:
            input_ = AttribDict()
            with open(filename) as fh:
                for line in fh:
                    # get rid of comments
                    line = line.split("#")[0].strip()
                    if not line:
                        continue
                    line = line.split()
                    input_category = input_.setdefault(line.pop(0), [])
                    input_item = AttribDict()
                    for item in line:
                        key, value = item.split("=", 1)
                        input_item[key] = _decode_string_value(value)
                    input_category.append(input_item)
        else:
            input_ = filename
        super(Inputfile, self).__init__(input_)
        self.get_Proj4()

    def get_Proj4(self):
        """
        Parse the ``grid`` line and figure out if a Proj4 projection
        was used.
        """
        proj_dict = AttribDict()
        try:
            proj_dict['proj'] = self.grid[0]['proj']
            proj = True
        except KeyError:
            # warn('No proj found... setting to ``utm``.')
            proj_dict['proj'] = 'utm'
            proj = False

        try:
            proj_dict['ellps'] = self.grid[0]['ellps']
            ellps = True
        except KeyError:
            if 'datum' in self.grid[0]:
                # warn('No ellps found, but datum was....')
                pass
            else:
                # warn('No ellps found... setting to ``WGS84``.')
                proj_dict['ellps'] = 'WGS84'
            ellps = False

        try:
            proj_dict['datum'] = self.grid[0]['datum']
            datum = True
        except KeyError:
            # warn('No datum found... '
            #      'setting to ``{}``.'.format(proj_dict['ellps']))
            proj_dict['datum'] = proj_dict['ellps']
            datum = False

        try:
            proj_dict['lon_0'] = self.grid[0]['lon_p']
            lon_p = True
        except KeyError:
            # warn('No lon_p found... '
            #      'setting to ``{}``.'.format(self.grid[0]['lon']))
            proj_dict['lon_0'] = self.grid[0]['lon']
            lon_p = False

        try:
            proj_dict['lat_0'] = self.grid[0]['lat_p']
            lat_p = True
        except KeyError:
            # warn('No lat_p found... '
            #      'setting to ``{}``.'.format(self.grid[0]['lat']))
            proj_dict['lat_0'] = self.grid[0]['lat']
            lat_p = False

        try:
            proj_dict['scale'] = self.grid[0]['scale']
            scale = True
        except KeyError:
            # warn('No scale found... setting to ``None``.')
            # proj_dict['scale'] = self.grid[0]['lat']
            scale = False

        if proj or ellps or datum or lon_p or lat_p or scale:
            self.is_proj4 = True
            self.proj4 = proj_dict
        else:
            self.is_proj4 = False
            self.proj4 = None

    def get_coordinates(self, key, xi=None, elev=None,
                        plane=0, coordinate=0,
                        distance=np.inf):
        """
        Gets coordinates for input keys that have 3D coordinates:

        **Catersian grid coordinates:**

        `x`, `y`, `z` (or `depth`)

        or in -

        **Geographical coordinates:**

        `lon`, `lat`, `depth` (or `z`)

        Parameters
        ----------
        key : str
            Keyword to look for.

        xi : array-like
            `x` coordinate along the cross-section

        elev : array-like
            Elevation of the top surface. Same size as `xi`.
            Used to correct the returned y-plotting coordinate if
            `depth` is encounterd.

        plane : int
            Indicates cross-section (``0`` or ``1``) or map (``2``).

        coordinate : float
            Plane coordinate.

        distance : float
            Threshold distance from the plane coordinate to include. By
            default everything is included but this can cause too many
            symbols to be plotted obscuring the image.

        Returns
        -------
        2-sequence
            x- and y-plotting coordinates.

        Examples
        --------
        >>> get_coordinates('source')

        for easy plotting with :meth:`~pySW4.postp.image.Image.plot`.
        """
        items = self.get(key, [])
        if not items:
            return None
        x = []
        y = []
        z = []

        for item in items:
            try:
                x_ = item.x
                y_ = item.y
                try:
                    z_ = item.z
                except AttributeError:
                    z_ = item.depth
            except AttributeError:
                # warn('NotImplementedError: ' + msg.format(key, item))
                # continue
                x_ = item.lon
                y_ = item.lat
                try:
                    z_ = item.z
                except AttributeError:
                    z_ = item.depth

                # geographical ===> cartesian coordinates
                x_, y_ = geographic2cartesian(
                    self.proj4, lon0=self.grid[0].lon, lat0=self.grid[0].lat,
                    az=self.grid[0].az, lon=x_, lat=y_,
                    m_per_lat=self.grid[0].get('mlat', 111319.5),
                    m_per_lon=self.grid[0].get('mlon'))

            try:
                x += [x_]
                y += [y_]
                z += [z_]
            except UnboundLocalError:
                continue
        if not x:
            return None

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        if plane == 0:
            nearest = nearest_values(x, coordinate, distance)
            x, y, z = x[nearest], y[nearest], z[nearest]
            idx = xi.searchsorted(y)
            z_cor = elev[idx] + z
            return xi[idx], z_cor
        elif plane == 1:
            nearest = nearest_values(y, coordinate, distance)
            x, y, z = x[nearest], y[nearest], z[nearest]
            idx = xi.searchsorted(x)
            z_cor = elev[idx] + z
            return xi[idx], z_cor
        elif plane == 2:
            nearest = nearest_values(z, coordinate, distance)
            x, y, z = x[nearest], y[nearest], z[nearest]
            return y, x


class Outputfile(AttribDict):
    """
    A container for simulation metadata parsed from the SW4 STDOUT
    saved to a file.

    **The keywords the parser looks for are:**

    - 'Grid' - Information about the grid discretization.
    - 'Receiver INFO' - Cartesian (x, y, z) coordinates of recievers placed with geographical (lon, lat, z or depth) coordinates.
    - 'Geographic and Cartesian' - corners of the computational grid.
    - 'Start Time' - of the simulation.
    - 'Goal Time' - time to simulate.
    - 'Number of time steps' - steps to compute.
    - 'dt' - size of each time step
    - 'Total seismic moment' - sum of all sources in the simulation.
    - 'Moment magnitude' - Moment magnitude based on :math:`M_0` .

    Parameters
    ----------
    filename : str or :class:`~obspy.core.util.attribdict.AttribDict`
        Path (relative or absolute) of an SW4 input file.
    """
    def __init__(self, filename):
        if type(filename) is str:
            output_ = AttribDict()

            grid = output_.setdefault('grid', [])
            reciever = output_.setdefault('reciever', [])
            corners = output_.setdefault('corners', [])

            fh = open(filename, 'r')
            while True:
                line = fh.readline()
                if not line:
                    fh.close()
                    break
                else:
                    line = line.strip()

                if 'reading input file' in line:
                    output_.read_input_file_phase = line.split(
                        'reading input file')[1]
                    continue

                if 'start up phase' in line:
                    output_.start_up_phase = line.split('start up phase')[1]
                    continue

                if 'solver phase' in line:
                    output_.solver_phase = line.split('solver phase')[1]
                    continue

                if line.startswith('Grid'):
                    line = line.split()
                    keys = line
                    while True:
                        line = fh.readline()
                        if line.startswith('Total'):
                            grid.append(
                                AttribDict(
                                    {'points': int(float(line.split()[-1]))}))
                            break
                        else:
                            line = line.split()
                            grid_i = AttribDict()
                            for k, v in zip(keys, line):
                                grid_i[k] = int(v)
                            grid.append(grid_i)

                if line.startswith('Receiver INFO'):
                    name = line.split()[-1][:-1]
                    while True:
                        line = fh.readline().strip()
                        if line.startswith('nearest'):
                            x, y, z = [
                                float(item) for item in line.split()[5:8]]
                            station = AttribDict(
                                {'station': name, 'x': x, 'y': y, 'z': z})
                            reciever.append(station)
                            break

                if line.startswith('Geographic and Cartesian'):
                    for i in range(4):
                        line = fh.readline().split(',')
                        number, lon = line[0].split(':')
                        line[0] = lon
                        corner = AttribDict({'number': number})
                        for item in line:
                            k, v = item.split('=')
                            corner[k.strip()] = float(v)
                        corners.append(corner)
                    continue

                if line.startswith('Start Time'):
                    start_time, goal_time = line.split('Goal Time =')

                    output_.start_time = float(start_time.split('=')[1])
                    output_.goal_time = float(goal_time)
                    continue

                if line.startswith('Number of time steps'):
                    npts, dt = line.split('dt:')

                    output_.npts = int(npts.split('=')[1])
                    output_.dt = float(dt)
                    continue

                if line.startswith('Total seismic moment'):
                    output_.M0 = float(line.split()[4])
                    continue

                if line.startswith('Moment magnitude'):
                    output_.Mw = float(line.split()[3])
        else:
            output_ = filename
        super(Outputfile, self).__init__(output_)


def read_metadata(inputfile, outputfile):
    """
    Function to read both SW4 input and output files at once.
    """
    return Inputfile(inputfile), Outputfile(outputfile)


def _decode_string_value(string_item):
    """
    Converts string representations of int/float to the
    corresponding Python type.

    Parameters
    ----------
    string_item: str
        Configuration value from SW4 input file in its string
        representation.

    Returns
    -------
    int or float or str
        Configuration value from SW4 input file as the correct
        Python type (bool values specified as ``0`` or ``1`` in SW4
        input file will still be of ``int`` type).
    """
    try:
        return int(string_item)
    except ValueError:
        pass
    try:
        return float(string_item)
    except ValueError:
        pass
    return string_item


def _parse_input_file_and_folder(input_file=None, folder=None):
    """
    Helper function to unify input location (or `None`) and output
    folder to work on.

    Use cases (in order of preference):

     * ``input_file="/path/to/input", folder=None``:
       input file is used for metadata and location of output folder
     * ``input_file="/path/to/input", folder="/path/to/output"``:
       input file is used for metadata, folder location is specified
       separately (make sure to not mismatch).
     * ``input_file=None, folder="/path/to/output"``:
       Do not use metadata from input (station locations etc. will not
       show up in plots) and only use output files from specified
       location.
    """
    if input_file is None and folder is None:
        msg = ("At least one of `input_file` or `folder` has to be "
               "specified.")
        raise ValueError(msg)

    if input_file:
        input_folder = os.path.dirname(os.path.abspath(input_file))
        input_ = Inputfile(input_file)
    else:
        input_ = None

    if input_:
        folder_ = os.path.join(input_folder, input_.fileio[0].path)
        if folder and os.path.abspath(folder) != folder_:
            msg = ("Both `input` and `folder` option specified. Overriding "
                   "folder found in input file ({}) with user specified "
                   "folder ({}).").format(folder_, folder)
            warn(msg)
        else:
            folder = folder_
    return input_, folder
