# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Helper module for specifying the material properties of the
#           computational domain.
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------
"""
- material_model.py -

Python module to help specifying the material properties of the computational
domain.

By: Shahar Shani-Kadmiel, September 2015, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import os
import numpy as np
from scipy.interpolate import UnivariateSpline


def get_vp(z, vp0=0.4):
    """Calculate a velocity gradient based on surface P-wave velocity
    `vp0` in km/s.

    returns vp in km/s"""

    vp = vp0

    return vp

def get_vs(vp):
    """Calculate Shear-wave velocity based on Brocher (2008).
    ``vp`` float or sequence, km/s

    returns vs in km/s"""

    vs = 0.7858-1.2344*vp+0.7949*vp**2-0.1238*vp**3+0.0064*vp**4
    return vs


def get_rho(vp):
    """Calculate density based on Brocher (2008).
    ``vp`` float or sequence, km/s

    returns rho in gr/cm^3"""

    rho = 1.6612*vp-0.4721*vp**2+0.0671*vp**3-0.0043*vp**4+0.000106*vp**5
    return rho


def get_qs(vs):
    """Calculate Shear Quality Factor based on Brocher (2008).
    ``vs`` float or sequence, km/s

    returns qs"""

    qs = -16+104.13*vs-25.225*vs**2+8.2184*vs**3
    try:
        qs[vs < 0.3] = 13
    except TypeError:
        if vs < 0.3:
            qs = 13

    return qs


def get_qp(qs):
    """Calculate Pressure Quality Factor based on Brocher (2008).
    ``qs`` float or sequence

    returns qp"""

    return 2*qs


def sample_func(x, y):
    return UnivariateSpline(x, y, k=1, s=0)


def read_Vfile(filename):
    """
    """
    header = ''
    depth = []
    vp = []; vs = []; rho = []
    qp = []; qs = []; gmf = []

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
                    for i,f in enumerate(fields[:-1]):
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


class V_model(object):
    """Velocity model object, has depth, vp, vs, rho, qp, qs"""
    def __init__(self, filename):

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

        self.calculate_missing_data()


    def calculate_missing_data(self):
        """Look for ``nans`` in the data and try to calculate
        based on Brocher (2008)"""
        if np.isnan(self.vs).any():
            indices = np.isnan(self.vs)
            self.vs[indices] = (get_vs(self.vp[indices]*1e-3)*1e3).round()
        if np.isnan(self.rho).any():
            indices = np.isnan(self.rho)
            self.rho[indices] = (get_rho(self.vp[indices]*1e-3)*1e3).round()
        if np.isnan(self.qs).any():
            indices = np.isnan(self.qs)
            self.qs[indices] = (get_qs(self.vs[indices]*1e-3)).round()
        if np.isnan(self.qp).any():
            indices = np.isnan(self.qp)
            self.qp[indices] = (get_qp(self.qs[indices])).round()


    def get_properties(self, depth, k=0):
        """ This function first fits the data in the velocity model
        with an interpolation function and then evaluates the properties
        at the requested depths. Interpolation can either be based on
        piecewise step functions (nearest) or on linear interpolation.

        Parameters
        -----------
        depth : sequence, int or float, of depths at which to evaluate
            the properties.

        k : int, 0 or 1, default: 0
            0 - gets the nearest value
            1 - linear interpolation between data points

        Returns:
        ---------
        vp, vs, rho, qp, qs"""

        properties = [self.vp, self.vs, self.rho, self.qp, self.qs]
        new_properties = []
        if k == 0:
            for p in properties:
                func = sample_func(self.depth, p)
                new_properties += [func(depth)]
        elif k ==1:
            for p in properties:
                func = sample_func(self.depth[::2], p[::2])
                new_properties += [func(depth)]

        return new_properties


    def get_depth(self, values, property='vp'):
        """ This function first fits the data in the velocity model
        with an interpolation function and then evaluates the depth
        corresponding to the value of the requested property.

        Interpolation is linear.

        Parameters
        -----------
        values : sequence, int or float, of the property for which to
            evaluate the depth.

        property : string, 'vp', 'vs', 'rho', 'qp', or 'qs'.
            default: 'vp'

        Returns:
        ---------
        depth"""

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

        func = sample_func(p[::2], self.depth[::2])
        depth = func(values)

        return depth


    def __str__(self):
        out = self.header
        for i in range(self.depth.size):
            out += ('%7d,%7d,%7d,%7d,%7d,%7d,%s\n'
                    %(self.depth[i], self.vp[i], self.vs[i],
                      self.rho[i], self.qp[i], self.qs[i], self.gmf[i]))
        return out


    def write2file(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(filename, 'w') as f:
            f.write(self.__str__())
            f.write('end\n')


def grid_spacing(vmin, fmax, ppw=15):
    """This function calculates the h parameter (grid_spacing)
    based on the requirement that the shortest wavelength (vmin/fmax)
    be sampled by a minimum points_per_wavelength (ppw) normally set
    to 15.
    """
    return int(vmin / (fmax * ppw))


def f_max(vmin, h, ppw=15):  # Note: ppw is regarded differently in WPP and SW4
    """Calculate teh maximum resolved frequency as a function of the
    minimum wave velocity, the grid spacing and the number of points
    per wavelength."""
    return float(vmin) / (h * ppw)


def f0(fmax, source_type):
    """Calculate the fundamental frequency f_0 based on fmax and the
    source type"""
    if source_type in ['Ricker', 'RickerInt', 'Gaussian', 'GaussianInt']:
        f_0 = fmax / 2.5
    elif source_type in ['Brune', 'BruneSmoothed']:
        f_0 = fmax / 4
    return f_0


def omega(f0, source_type):
    """Calculate omega, that value that goes on the source line in the
    WPP input file as ``freq`` based on f_0 and the source type"""
    if source_type in ['Ricker', 'RickerInt']:
        freq = f0
    elif source_type in ['Brune', 'BruneSmoothed', 'Gaussian', 'GaussianInt']:
        freq = f0 * 2 * np.pi
    return freq


def get_vmin(h, fmax, ppw=15):
    return h * fmax * ppw


def get_z(v, v0, v_grad):
    return (v - v0) / v_grad
