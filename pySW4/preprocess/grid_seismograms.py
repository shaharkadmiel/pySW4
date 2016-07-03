from __future__ import absolute_import, print_function, division

from shutil import copyfile
import numpy as np

def grid_seismograms(x_extent, y_extent, mode='displacement',
                     x_n_of_stations=11, y_n_of_stations=11,
                     x_spacing=None, y_spacing=None,
                     xmin=None, xmax=None,
                     ymin=None, ymax=None,
                     z=0.,
                     infile=None, name='grid', writeEvery=100):

    """
    Setup a grid of stations.

    Takes the x,y extens of the grid and outputs a string for the infile
    If an infile name is given the function copys the file,appends '+traces' to it's name
    and appends the string to the file
    IMPORTANT - if you want to use this option more then once make sure to change the filename after the first time

    It is possible to set the extent of the recivers grid using the xmin,xmax,ymin,ymax args
    If none are given the first and last recivers will be located at simulation bounderies
    NOTE! that they are probably located on the damping layer
    It is also possible to get stations along one line by setting xmin=xmax or ymin=ymax

    Deafult number of stations is 11 in each direction,
    It is possible to set the number of stations OR the spacing between them
    If you specify the spacing the function might slightly change them  so that
    a whole number of stations matches the extent of the grid

    mode can be 'displacement' (solution), 'velocity' (differenation of the solution),div,curl,or strains
    Notice spelling should match sw4 infile syntax

    The depth z of the grid could also be set and is deafult to 0
    """

    if xmin is None: xmin=0.
    if ymin is None: ymin=0.
    if xmax is None: xmax=x_extent
    if ymax is None: ymax=y_extent

    xmax=np.minimum(xmax,x_extent)
    ymax=np.minimum(ymax,y_extent)

    if xmin==xmax: x_n_of_stations=1
    if ymin==ymax: y_n_of_stations=1

    if x_spacing is not(None): x_n_of_stations=(xmax-xmin)/(x_spacing)+1
    if y_spacing is not(None): y_n_of_stations=(ymax-ymin)/(y_spacing)+1

    x_n_of_stations=round(x_n_of_stations)
    y_n_of_stations=round(y_n_of_stations)

    x=np.linspace(xmin,xmax,x_n_of_stations)
    y=np.linspace(ymin,ymax,y_n_of_stations)
    sac_string = 'rec x=%.3f y=%.3f depth=%.3f file=%s_x=%.3f_y=%.3f_ writeEvery=%d variables=%s\n'

    string = "\n\n#-----------------%d seismograms added for grid: %s -------------------\n\n" %(len(x)*len(y), name)

    for i in x:
        for j in y:
            string += (sac_string
                %(i,j,z,name,i,j,writeEvery,mode))

    if infile is None:
        return string
    else:
        try:
            if 'traces' not in infile:
                filename,extention = infile.rsplit('.',1)
                filename += '+traces.' + extention
                copyfile(infile, filename)
            else:
                filename = infile
        except IOError:
            filename=infile

        with open(filename, "a") as f:
            f.write(string)
        f.close()

        return filename
