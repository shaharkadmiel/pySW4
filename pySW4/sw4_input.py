# -*- coding: utf-8 -*-
"""
Parsing routines for SW4 input files and directories.

.. module:: sw4_input

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
"""
from __future__ import absolute_import, print_function, division

import os
from warnings import warn
from obspy.core.util import AttribDict
import numpy as np
from .utils import nearest_values


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

    def get_coordinates(self, key, xi=None, elev=None,
                        plane=0, coordinate=0,
                        distance=np.inf):
        """
        Gets coordinates for input keys that have 3D coordinates:

        **Catersian grid coordinates:**

        ``x``, ``y``, ``z`` (or ``depth``)

        or in -

        **Geographical coordinates:**

        ``lon``, ``lat``, ``depth`` (or ``z``)

        Parameters
        ----------
        key : str
            Keyword to look for.

        xi : sequence
            ``x`` coordinate along the cross-section

        elev : sequence
            Elevation of the top surface. Same size as ``xi``.
            Used to correct the returned y-plotting coordinate if
            ``depth`` is encounterd.

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

        Example
        -------
        >>> get_coordinates('source')

        for easy plotting with :meth:`~pySW4.postp.image.Image.plot`.

        Note
        ----
        Geographical to Cartesian grid coordinates is not yet
        implemented. You are more than welcome to contribute.
        """
        items = self.get(key, [])
        if not items:
            return None
        x = []
        y = []
        z = []
        msg = ('`{}` was given in geographical coordinates, '
               'transforming those into cartesian grid coordinates is '
               'not implemented yet:\n'
               '{}')
        for item in items:
            try:
                x_ = item.x
                y_ = item.y
                try:
                    z_ = item.z
                except AttributeError:
                    z_ = item.depth
            except AttributeError:
                warn('NotImplementedError: ' + msg.format(key, item))
                continue
                x_ = item.lon
                y_ = item.lat
                try:
                    z_ = item.z
                except AttributeError:
                    z_ = item.depth
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
            z_cor = elev[idx] - z
            return xi[idx], z_cor
        elif plane == 1:
            nearest = nearest_values(y, coordinate, distance)
            x, y, z = x[nearest], y[nearest], z[nearest]
            idx = xi.searchsorted(x)
            z_cor = elev[idx] - z
            return xi[idx], z_cor
        elif plane == 2:
            nearest = nearest_values(z, coordinate, distance)
            x, y, z = x[nearest], y[nearest], z[nearest]
            return x, y


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
        input_ = parse_input_file(input_file)
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
