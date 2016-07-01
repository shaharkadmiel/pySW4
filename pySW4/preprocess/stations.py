# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Placing stations to record synthetic seismograms on SW4
#           simulations
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------

"""
- stations.py -

Python module for placing stations for recording synthetics in SW4
simulations.

By: Shahar Shani-Kadmiel, September 2015, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import os, shutil


def line_of_stations(x1=None    ,x2=None,
                     y1=None    ,y2=None,
                     lon1=None  ,lon2=None,
                     lat1=None  ,lat2=None,
                     depth1=0   ,depth2=0,
                     number_of_stations=3, name='line', mode='displacement',
                     writeEvery=100, nswe=0, simulator='sw4',
                     infile=None):
    """This function places stations to record synthetics on a line:

    Params:
    --------

    (x1   , y1   , depth1) and (x2   , y2   , depth2) or
    (lon1 , lat1 , depth1) and (lon2 , lat2 , depth2) are the ends
    of the line

    number_of_stations : Number of stations to place on the line (defaults to 3)

    name : prefix of the sac file name

    mode : 'displacement' (default), 'velocity', 'div', 'curl', or 'strains'

    writeEvery : cycle interval to write the data to disk

    nsew : the components of the station, default is 0 (x,y,z) but can be set to
        1 (East,North,Vertical)

    simulator : 'sw4' (default, or 'wpp')

    infile : if not None stations are added to the a copy of the specified file
        and saved with the string '+name' where name is line (by default).
        Otherwise, a formated string id printed to stdout which can then be
        copied to the input file.
    """

    if x1 is not None and x2 is not None:
        x = np.linspace(x1, x2, number_of_stations)
        if simulator is 'sw4':
            sac_string = 'rec x=%.3f y=%.3f depth=%.3f file=%s_x=%.3f_y=%.3f_z=%.3f__x=%.3f_y=%.3f_z=%.3f_ writeEvery=%d variables=%s\n'
        elif simulator is 'wpp':
            sac_string = 'sac x=%.3f y=%.3f depth=%.3f file=%s_x=%.3f_y=%.3f_z=%.3f_ writeEvery=%d velocity=%d variables=%s\n'


    elif lon1 is not None and lon2 is not None:
        x = np.linspace(lon1, lon2, number_of_stations)
        if simulator is 'sw4':
            sac_string = 'rec lon=%.10f lat=%.10f depth=%.3f file=%s_lon=%.10f_lat=%.10f_depth=%.3f_ writeEvery=%d variables=%s\n'
        elif simulator is 'wpp':
            sac_string = 'sac lon=%.10f lat=%.10f depth=%.3f file=%s_lon=%.10f_lat=%.10f_depth=%.3f_ writeEvery=%d velocity=%d variables=%s\n'


    if y1 is not None and y2 is not None:
        y = np.linspace(y1, y2, number_of_stations)
    elif lat1 is not None and lat2 is not None:
        y = np.linspace(lat1, lat2, number_of_stations)

    elif depth1 is not None and depth2 is not None:
        z = np.linspace(depth1, depth2, number_of_stations)

    string = "\n\n#------------------- stations on a line: %s -------------------\n" %name
    if simulator is 'sw4':
        for i in range(len(x)):
            string += (sac_string %(x[i],y[i],z[i],
                                    name,x[i],y[i],z[i],writeEvery,mode))

    elif simulator is 'wpp':
        if mode is 'displacement':
            velocity = 0
            mode = 'solution'
        elif mode is 'velocity':
            velocity = 1
            mode = 'solution'
        elif mode in ('curl', 'div', 'strains'):
            velocity = 0
        for i in range(len(x)):
            string += (sac_string %(x[i],y[i],z[i],
                                    name,x[i],y[i],z[i],writeEvery,
                                    velocity,mode))

    if infile is None:
        return string
    else:
        # copy the premade input file,
        # add '+name' to the filename
        # and open the new file in append mode
        filename,extention = os.path.splitext(infile)
        filename += '+' + name + extention
        shutil.copyfile(infile, filename)

        with open(filename, "a") as f:
            f.write(string)
        f.close()

        return filename

def place_station(x=None, y=None, lat=None, lon=None, depth=0,
                  name='st', mode='displacement',
                  writeEvery=100, nswe=0, simulator='sw4',
                  infile=None):
    """This function places synthetic stations at locations (x,y) in grid
    coordinates or at (lat, lon) in geographical coordinates.
    A sequence (list or tuple) of station names can be passed on to name.
    velocity and writeEvery are SW4 parameters, see SW4 User Guide for more
    info.

    If infile is None, the function returns a formatted string which can be
    copied later to an inputfile. If a SW4 inputfile name is passed, the
    formated lines are appended at the end of the file and the string '+st'
    is appended to the file name if not already there. The original inputfile
    is unaffaected.
    """

    if x is not None and y is not None:
        if simulator is 'sw4':
            sac_string = 'rec x=%.3f y=%.3f depth=%.3f file=%s writeEvery=%d variables=%s\n'
        elif simulator is 'wpp':
            sac_string = 'sac x=%.3f y=%.3f depth=%.3f file=%s writeEvery=%d velocity=%d variables=%s\n'
    elif lon is not None and lat is not None:
        x,y = lon,lat
        if simulator is 'sw4':
            sac_string = 'rec lon=%.10f lat=%.10f depth=%.3f file=%s writeEvery=%d variables=%s\n'
        elif simulator is 'wpp':
            sac_string = 'sac lon=%.10f lat=%.10f depth=%.3f file=%s writeEvery=%d velocity=%d variables=%s\n'

    string = "\n\n#------------------- stations at locations: -------------------\n"
    try:
        if simulator is 'sw4':
            for i in range(len(x)):
                string += (sac_string %(x[i],y[i],depth,
                           name[i],writeEvery,mode))

        elif simulator is 'wpp':
            if mode is 'displacement':
                velocity = 0
                mode = 'solution'
            elif mode is 'velocity':
                velocity = 1
                mode = 'solution'
            elif mode in ('curl', 'div', 'strains'):
                velocity = 0
            for i in range(len(x)):
                string += (sac_string
                            %(x[i],y[i],depth,
                            name[i],writeEvery,velocity,mode))

    except TypeError:
        if simulator is 'sw4':
            for i in range(len(x)):
                string += (sac_string %(x,y,depth,
                           name[i],writeEvery,mode))

        elif simulator is 'wpp':
            if mode is 'displacement':
                velocity = 0
                mode = 'solution'
            elif mode is 'velocity':
                velocity = 1
                mode = 'solution'
            elif mode in ('curl', 'div', 'strains'):
                velocity = 0
            for i in range(len(x)):
                string += (sac_string
                            %(x,y,depth,
                            name[i],writeEvery,velocity,mode))

    if infile is None:
        return string
    else:
        # copy the premade input file,
        # add '+st' to the filename
        # and open the new file in append mode
        if 'st' not in infile:
            filename,extention = os.path.splitext(infile)
            filename += '+st' + extention
            shutil.copyfile(infile, filename)
        else:
            filename = infile

        with open(filename, "a") as f:
            f.write(string)
        f.close()

        return filename
