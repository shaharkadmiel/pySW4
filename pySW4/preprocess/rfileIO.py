# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Reading and writing rfiles.
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------

"""
- rfile.py -

Python module to read and write rfiles.

By: Shahar Shani-Kadmiel, September 2015, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import os, sys
import numpy as np

flush = sys.stdout.flush()

hdr_dtype = np.dtype([
        ('magic'       , 'int32'   ),
        ('precision'   , 'int32'   ),
        ('attenuation' , 'int32'   ),
        ('az'          , 'float64' ),
        ('lon0'        , 'float64' ),
        ('lat0'        , 'float64' ),
        ('mlen'        , 'int32'   )
    ])

block_hdr_dtype = np.dtype([
        ('hh'  ,  'float64' ),
        ('hv'  ,  'float64' ),
        ('z0'  ,  'float64' ),
        ('nc'  ,  'int32'   ),
        ('ni'  ,  'int32'   ),
        ('nj'  ,  'int32'   ),
        ('nk'  ,  'int32'   )
    ])

def write_hdr(f, magic=1, precision=4, attenuation=1,
             az=0., lon0=33.5, lat0=28.0,
             proj_str='+proj=longlat +datum=WGS84 +no_defs', nb=1):

    """Write rfile header"""

    magic        = np.int32(magic)
    precision    = np.int32(precision)
    attenuation  = np.int32(attenuation)
    az           = np.float64(az)
    lon0         = np.float64(lon0)
    lat0         = np.float64(lat0)
    mlen         = np.int32(len(proj_str))
    nb           = np.int32(nb)

    hdr = [magic,precision,attenuation,az,lon0,lat0,mlen,proj_str,nb]
    for val in hdr:
        f.write(val)
    return


def read_hdr(f):
    (magic,precision,
     attenuation,az,lon0,lat0,mlen) = np.fromfile(f, hdr_dtype, 1)[0]
    proj_str_dtype = 'S' + str(mlen)
    proj_str = np.fromfile(f, proj_str_dtype, 1)[0]
    nb = np.fromfile(f, 'int32', 1)[0]

    return magic,precision,attenuation,az,lon0,lat0,mlen,proj_str,nb


def write_block_hdr(f, hh,hv,z0,nc,ni,nj,nk):

    """Write rfile block header"""

    hh = np.float64(hh)
    hv = np.float64(hv)
    z0 = np.float64(z0)
    nc = np.int32(nc)
    ni = np.int32(ni)
    nj = np.int32(nj)
    nk = np.int32(nk)

    block_hdr = [hh,hv,z0,nc,ni,nj,nk]
    for val in block_hdr:
        f.write(val)


def read_block_hdr(f):
    hh,hv,z0,nc,ni,nj,nk = np.fromfile(f, block_hdr_dtype, 1)[0]

    return hh,hv,z0,nc,ni,nj,nk

def read(filename, verbose=False):
    """
    Read rfile into Model object
    """

    model = Model()
    model.filename = filename

    with open(filename, 'rb') as f:
        (model.magic,
         model.precision,
         model.attenuation,
         model.az,
         model.lon0,
         model.lat0,
         model.mlen,
         model.proj_str,
         model.nb) = read_hdr(f)

        if verbose:
            print(model)
            flush

        for b in range(model.nb):
            block = Block()
            block.number = b+1

            (block.hh,
             block.hv,
             block.z0,
             block.nc,
             block.ni,
             block.nj,
             block.nk) = read_block_hdr(f)

            model.blocks += [block]
            if verbose:
                print(block)
                flush

        for b in range(model.nb):
            hh = model.blocks[b].hh
            hv = model.blocks[b].hv
            z0 = model.blocks[b].z0
            nc = model.blocks[b].nc
            ni = model.blocks[b].ni
            nj = model.blocks[b].nj
            nk = model.blocks[b].nk

            z = np.fromfile(f, np.float32, ni*nj*nk*nc)

            if nc == 1: # topo is independant of k
                model.blocks[b].data = z.reshape(ni,nj)
            else:
                # C-order reshape
                model.blocks[b].data = z.reshape(ni,nj,nk,nc)

            model.blocks[b].xyextent = (0                  , (nj-1)*hh*1e-3,
                                        0                  , (ni-1)*hh*1e-3)
            model.blocks[b].xzextent = (0                  , (ni-1)*hh*1e-3,
                                        (z0+(nk-1)*hv)*1e-3, z0*1e-3       )
            model.blocks[b].yzextent = (0                  , (nj-1)*hh*1e-3,
                                        (z0+(nk-1)*hv)*1e-3, z0*1e-3       )

    return model

class Model(object):
    """
    A class to hold rfile header and blocks
    """

    def __init__(self):
        self.filename       = None
        self.precision      = 4
        self.attenuation    = 1
        self.az             = 0.0
        self.lon0           = 33.5
        self.lat0           = 28.0
        self.proj_str       = '+proj=longlat +datum=WGS84 +no_defs'
        self.nb             = 1

        self.blocks = []

    def __str__(self):
        string = '\nModel information:\n'
        string +=  '-----------------\n'
        string +=  '        Filename : %s\n' %self.filename
        string +=  '           lon 0 : %s\n' %self.lon0
        string +=  '           lat 0 : %s\n' %self.lat0
        string +=  '         Azimuth : %s\n' %self.az
        string +=  '    Proj4 string : %s\n' %self.proj_str
        string +=  'Number of blocks : %s\n' %self.nb

        return string


class Block(object):
    """
    A class to hold rfile block data and header
    """

    def __init__(self):
        self.number       = 0
        self.hh           = 0.0
        self.hv           = 0.0
        self.z0           = 0.0
        self.nc           = 1
        self.ni           = 0
        self.nj           = 0
        self.nk           = 1

        self.xyextent = ()
        self.xzextent = ()
        self.yzextent = ()

        self.data         = None

    def __str__(self):
        string =   'Block information:\n'
        string +=  '-----------------\n'
        string +=  '          Number : %s\n' %self.number
        string +=  ' Horizontal h, m : %s\n' %self.hh
        string +=  '   Vertical h, m : %s\n' %self.hv
        string +=  '          z 0, m : %s\n' %self.z0
        string +=  '              ni : %s\n' %self.ni
        string +=  '              nj : %s\n' %self.nj
        string +=  '              nk : %s\n' %self.nk
        string +=  '              nc : %s\n' %self.nc

        return string

    def vp(self):
        """ Return values of vp for block"""
        return self.data[:,:,:,1]


    def vs(self):
        """ Return values of vs for block"""
        return self.data[:,:,:,2]


    def rho(self):
        """ Return values of density for block"""
        return self.data[:,:,:,0]


    def qp(self):
        """ Return values of Qp for block"""
        return self.data[:,:,:,3]


    def qs(self):
        """ Return values of Qs for block"""
        return self.data[:,:,:,4]




