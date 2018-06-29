# -*- coding: utf-8 -*-
"""
Python module to help specify the material properties and the grid
parameters of the computational domain.

.. module:: material_model

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
import numpy as np
from scipy.interpolate import UnivariateSpline


def get_vs(vp):
    """
    Calculate Shear-wave velocity based on Brocher (2008).

    .. note::
        Shear-wave velocity is forced to be greater than
        :math:`V_P / \\sqrt{2}`.

    Parameters
    ----------
    vp : float or sequence
        Pressure-wave velocity in km/s.

    Returns
    -------
    float or sequence
        Shear-wave velocity in km/s.
    """
    # vs = (0.7858
    #       - 1.2344 * vp
    #       + 0.7949 * vp**2
    #       - 0.1238 * vp**3
    #       + 0.0064 * vp**4)

    # vs[vp <= 2] = vp[vp <= 2] / 1.73

    vs = vp / 1.73

    # try:
    #     vs[vp / vs < 1.42] = vp[vp / vs < 1.42] / 1.4
    # except TypeError:
    #     if vp / vs < 1.42:
    #         print(vs)
    #         vs = vp / 1.4
    #         print(vs)
    return vs


def get_rho(vp):
    """
    Calculate :math:`\\rho` (density) based on Brocher (2008).

    Parameters
    ----------
    vp : float or sequence
        Pressure-wave velocity in km/s.

    Returns
    -------
    float or sequence
        :math:`\\rho` (density) in gr/cm^3.
    """
    rho = (1.6612 * vp
           - 0.4721 * vp**2
           + 0.0671 * vp**3
           - 0.0043 * vp**4
           + 0.000106 * vp**5)
    return rho


def get_qs(vs):
    """
    Calculate Shear-wave Quality Factor based on Brocher (2008).

    .. note:: If Shear-wave velocity is less-than 0.3 km/s, Shear-wave
              Quality Factor is set to 13.

    Parameters
    ----------
    vs : float or sequence
        Shear-wave velocity in km/s.

    Returns
    -------
    float or sequence
        Shear-wave Quality Factor.
    """
    qs = (-16
          + 104.13 * vs
          - 25.225 * vs**2
          + 8.2184 * vs**3)
    try:
        qs[vs < 0.3] = 13
    except TypeError:
        if vs < 0.3:
            qs = 13

    return qs


def get_qp(qs):
    """Calculate Pressure-wave Quality Factor based on Brocher (2008).

    Parameters
    ----------
    qs : float or sequence
        Shear-wave Quality Factor.

    Returns
    -------
    float or sequence
        Pressure-wave Quality Factor.
    """
    return 2 * qs


def _sample_func(x, y):
    """
    Helper function to interpolate between discrete ``x, y`` points.

    Used by the following methods
    - :meth:`~.V_model.get_properties`
    and
    - :meth:`~.V_model.get_depth`
    """
    return UnivariateSpline(x, y, k=1, s=0)


def read_Vfile(filename):
    """
    Read a Velocity Model file.

    Vfile format has a flexible header (line starting with a '#'
    character) and a *comma separated values* data section where each
    line has *Depth*, *Vp*, *Vs*, *rho*, *Qp*, *Qs*, *Grp./Form./Mbr.*
    values and ends with a line containing one word 'end'. The values
    in each line are "true" to that specific depth-point.

    **Example Vfile:**
    ::

        # Just a simple made up velocity model
        # Shani-Kadmiel (2016)
        # Depth| Vp    | Vs    | rho   | Qp    | Qs    | Grp/Form./Mbr.
        #   m  | m/s   | m/s   | kg/m^3|
              0,   1000,    500,   1000,     20,     10, Upper made up
           2000,   1500,    750,   1200,     30,     15, Middle made up
           5000,   2000,   1000,   1600,     70,     35, Lower made up
        end

    Parameters
    ----------
    filename : str
        Path (relative or absolute) to the Vfile.

    Returns
    -------
    array-like
        A list of header, depth, vp, vs, rho, qp, qs, gmf values.

    See also
    --------
    .V_model
    """
    header = ''
    depth = []
    vp = []
    vs = []
    rho = []
    qp = []
    qs = []
    gmf = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header += line
            elif line.startswith('\n'):
                continue
            elif line.startswith('end'):
                break
            else:
                try:
                    fields = line.split(',')
                    for i, f in enumerate(fields[:-1]):
                        try:
                            fields[i] = float(f)
                        except ValueError:
                            fields[i] = np.nan

                    ldepth, lvp, lvs, lrho, lqp, lqs, lgmf = fields
                    depth += [ldepth]
                    vp    += [lvp]
                    vs    += [lvs]
                    rho   += [lrho]
                    qp    += [lqp]
                    qs    += [lqs]
                    gmf   += [lgmf[:-1]]
                except ValueError:
                    continue

    depth = np.array(depth, dtype=float)
    vp    = np.array(vp,    dtype=float)
    vs    = np.array(vs,    dtype=float)
    rho   = np.array(rho,   dtype=float)
    qp    = np.array(qp,    dtype=float)
    qs    = np.array(qs,    dtype=float)

    return header, depth, vp, vs, rho, qp, qs, gmf


class V_model():
    """
    Class to handle Velocity models.

    Uses :func:`~.read_Vfile` function to read Vfile format files.

    Parameters
    ----------
    filename : str
        Path (relative or absolute) to the Vfile.

    calculate_missing_data : bool
        If ``True`` (default), any missing data is calculated based on
        Brocher (2008).
    """
    def __init__(self, filename, calculate_missing_data=True):
        self.name = os.path.splitext(os.path.split(filename)[-1])[0]
        self.filename = os.path.abspath(filename)

        (self.header,
         self.depth,
         self.vp,
         self.vs,
         self.rho,
         self.qp,
         self.qs,
         self.gmf) = read_Vfile(self.filename)

        if calculate_missing_data:
            self._calculate_missing_data()

    def _calculate_missing_data(self):
        """
        Looks for ``nan`` s in the data and trys to calculate and fill
        based on Brocher (2008).
        """
        if np.isnan(self.vs).any():
            indices = np.isnan(self.vs)
            self.vs[indices] = (
                get_vs(self.vp[indices] * 1e-3) * 1e3).round()

        if np.isnan(self.rho).any():
            indices = np.isnan(self.rho)
            self.rho[indices] = (
                get_rho(self.vp[indices] * 1e-3) * 1e3).round()

        if np.isnan(self.qs).any():
            indices = np.isnan(self.qs)
            self.qs[indices] = (
                get_qs(self.vs[indices] * 1e-3)).round()

        if np.isnan(self.qp).any():
            indices = np.isnan(self.qp)
            self.qp[indices] = (
                get_qp(self.qs[indices])).round()

    def get_properties(self, depth, k=0):
        """
        This function first fits the data in the velocity model
        with an interpolation function using :func:`~._sample_func` and
        then evaluates the properties at the requested ``depths``.
        Interpolation can either be based on piecewise step functions
        that uses the nearest value or on a linear interpolation.

        Parameters
        ----------
        depth : sequence or int or float
            Depths at which to evaluate the properties.

        k : int
            Interpolation method:

            - 0 - nearest value (default)
            - 1 - linear interpolation between data points

        Returns
        -------
        array-like
            vp, vs, rho, qp, qs
        """

        properties = [self.vp, self.vs, self.rho, self.qp, self.qs]
        new_properties = []
        if k == 0:
            for p in properties:
                func = _sample_func(self.depth, p)
                new_properties += [func(depth)]
        elif k == 1:
            for p in properties:
                func = _sample_func(self.depth[::2], p[::2])
                new_properties += [func(depth)]

        return new_properties

    def get_depth(self, values, property='vp'):
        """
        This function linearly interpolats the data in the velocity
        model and then evaluates the depth corresponding to the value of
        the requested property.

        Parameters
        ----------
        values : sequence or int or float
            The property for which to evaluate the depth of.

        property : {'vp' (default), 'vs', 'rho, 'qp', 'qs'}
            Property corresponding to `value`.

        Returns
        -------
        sequence or float
            Depth of the requested property.
        """

        if property is 'vp':
            p = self.vp
        elif property is 'vs':
            p = self.vs
        elif property is 'rho':
            p = self.rho
        elif property is 'qp':
            p = self.qp
        elif property is 'qs':
            p = self.qs

        func = _sample_func(p[::2], self.depth[::2])
        depth = func(values)

        return depth

    def __str__(self):
        out = self.header
        for i in range(self.depth.size):
            out += ('%7d,%7d,%7d,%7d,%7d,%7d,%s\n'
                    % (self.depth[i], self.vp[i], self.vs[i], self.rho[i],
                       self.qp[i], self.qs[i], self.gmf[i]))
        return out

    def write2file(self, filename=None):
        """
        Save Vfile for reading later into :class:`~.Vfile` class.
        """
        if filename is None:
            filename = self.filename
        with open(filename, 'w') as f:
            f.write(self.__str__())
            f.write('end\n')


def grid_spacing(vmin, fmax, ppw=8):
    """
    Calculate the ``h`` parameter (``grid_spacing``) based on the
    requirement that the shortest wavelength
    (:math:`\\lambda=V_{min}/f_{max}`) be sampled by a minimum
    points-per-wavelength (``ppw``).

    Parameters
    ----------
    vmin : float
        Minimum seismic velocity in the computational doamin in m/s.

    fmax : float
        Maximum frequency in the source-time function frequency content,
        Hz.

    ppw :
        Minimum points-per-wavelenght required for the computation. Set
        to 8 by default.

    Returns
    -------
    float
        The suggested grid spacing in meters.

    See also
    --------
    .get_vmin
    .source_frequency
    .f_max
    """
    return float(vmin) / (fmax * ppw)


def get_vmin(h, fmax, ppw=8):
    """
    Calculate the minimum allowed velocity that meets the requirement
    that the shortest wavelength (:math:`\\lambda=V_{min}/f_{max}`) be
    sampled by a minimum points-per-wavelength (`ppw`).

    Parameters
    ----------
    h : float
        Grid spacing of the computational doamin in meters.

    fmax : float
        Maximum frequency in the source-time function frequency content,
        Hz.

    ppw :
        Minimum points-per-wavelenght required for the computation. Set
        to 8 by default.

    Returns
    -------
    float
        The suggested grid spacing in meters.

    See also
    --------
    .grid_spacing
    .source_frequency
    .f_max
    """
    return h * fmax * ppw


# def get_z(v, v0, v_grad):
#     return (v - v0) / v_grad
