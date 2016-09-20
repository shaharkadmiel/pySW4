# -*- coding: utf-8 -*-
"""
Convert a sequencial set of .png images to a .mp4 movie

.. module:: plot_image

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import, print_function, division

import os
import sys
import argparse
from ..postp.image import read_image


def main(argv=None):

    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: {}\n'.format(message))
            self.print_help()
            sys.exit(2)

    parser = DefaultHelpParser(
        description='Plot all (or specific) patches in an .sw4img file.'
    )

    parser.add_argument('-patches', action='store', default='None',
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

    parser.add_argument('-no_cb', action='store_true',
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

    image = read_image(args.imagefile, args.stf)

    backend = plt.get_backend()
    try:
        plt.switch_backend('AGG')
        if args.patches is not None:
            patches = args.patches.split()

        fig = image.plot(patches, vmin=args.vmin, vmax=args.vmax,
                         colorbar=args.no_cb, cmap=args.cmap)
        name, _ = os.path.splitext(os.path.basename(args.imagefile))
        fig.savefig(os.path.join((args.save_path, name + '.' + args.format)),
                    bbox_inches='tight', dpi=args.dpi)
    finally:
        plt.switch_backend(backend)

if __name__ == "__main__":
    main()
