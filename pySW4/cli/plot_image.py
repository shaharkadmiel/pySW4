# -*- coding: utf-8 -*-
"""
Plot a single SW4 image.

.. rubric:: Instructions:

This module can be called from the command line to plot a single SW4
image. It can be usefull to run this on the server-end to generate
quick-and-dirty plots for *pseudo-RunTime* visualization of the results.
Wrap it in a *cronjob* that plots the image from the last timestep and
you are set.

Type ``pySW4-plot-image`` to get the help message.

**Example**
::

    $ pySW4-plot-image -cmap jet \\
        -format png 'results/berkeley.cycle=00000.z=0.p.sw4img'

will save the SW4 image from the Berkeley rfile example,
``berkeley.cycle=00000.z=0.p.sw4img`` to a *.png* image,
``./berkeley.cycle=00000.z=0.p.png`` with the ``jet`` colormap.

.. module:: plot_image

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import, print_function, division

import os
import sys
import argparse
from matplotlib.pyplot import get_backend, switch_backend
from pySW4.postp.image import read_image


def main(argv=None):

    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: {}\n'.format(message))
            self.print_help()
            sys.exit(2)

    parser = DefaultHelpParser(
        prog='pySW4-plot-image',
        description='Plot all (or specific) patches in an .sw4img file.'
    )

    parser.add_argument('-patches', action='store', default=None,
                        help='Patches to plot. If None, all patches are '
                             'plotted. Otherwise provide a sequence of '
                             'integers in quotation marks. For example: '
                             '"0 1 3" will plot patches 0, 1, 3. (default: '
                             '%(default)s)',
                        metavar='None')

    parser.add_argument('-stf', action='store', default='displacement',
                        help='Source-time function type used to initiate '
                             'wave propagation in the simulation. '
                             'displacement or velocity. (default: '
                             '%(default)s)',
                        metavar='displacement')

    parser.add_argument('-vmin', action='store', default=None,
                        help='Manually set minimum of color scale. (default: '
                             '%(default)s)',
                        metavar='None')

    parser.add_argument('-vmax', action='store', default=None,
                        help='Manually set maximum of color scale. (default: '
                             '%(default)s)',
                        metavar='None')

    parser.add_argument('-no_cb', action='store_false',
                        help='Suppress colorbar along with the image.')

    parser.add_argument('-cmap', action='store', default=None,
                        help='Matplotlib colormap. If None, a suitable '
                             'colormap is automatically picked (default: '
                             '%(default)s)',
                        metavar=None)

    parser.add_argument('-dpi', action='store', default=300,
                        help='Resolution of the saved image (default: '
                             '%(default)s)',
                        metavar=300)

    parser.add_argument('-format', action='store', default='pdf',
                        help='Image format to save. {pdf, png, jpeg, ...} '
                             '(default: %(default)s)',
                        metavar='pdf')

    parser.add_argument('-save_path', action='store', default='./',
                        help='Path (relative or absolute) to where image is '
                             'saved.(default: %(default)s)',
                        metavar='./')

    parser.add_argument('imagefile', action='store',
                        help='Path (relative or absolute) to the SW4 image.')
    args = parser.parse_args(argv)

    image = read_image(args.imagefile, stf=args.stf)

    backend = get_backend()
    try:
        switch_backend('AGG')
        if args.patches is not None:
            patches = [int(item) for item in args.patches.split()]
        else:
            patches = args.patches

        try:
            vmin = float(args.vmin)
        except TypeError:
            vmin = args.vmin

        try:
            vmax = float(args.vmax)
        except TypeError:
            vmax = args.vmax

        fig = image.plot(patches, vmin=vmin, vmax=vmax,
                         colorbar=args.no_cb, cmap=args.cmap)[0]
        name, _ = os.path.splitext(os.path.basename(args.imagefile))
        name = os.path.join(args.save_path, name + '.' + args.format)

        fig.savefig(name, bbox_inches='tight', dpi=args.dpi)
        print('Saved {} ===> {}.'.format(args.imagefile, name))
    finally:
        switch_backend(backend)

if __name__ == "__main__":
    main()
