# -*- coding: utf-8 -*-
"""
Convert a sequencial set of .png images to a .mp4 movie

.. module:: png2mp4

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

import sys
import argparse
from pySW4.plotting.png2mp4 import png2mp4


def main(argv=None):

    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: {}\n'.format(message))
            self.print_help()
            sys.exit(2)

    parser = DefaultHelpParser(
        prog='png2mp4',
        description='Convert a sequencial set of '
                    '.png images to a .mp4 movie.')

    parser.add_argument('-crf', action='store', default=23, type=int,
                        help='Constant Rate Factor, ranges from 0 to 51: A '
                             'lower value is a higher quality and vise '
                             'versa. The range is exponential, so increasing '
                             'the CRF value +6 is roughly half the bitrate '
                             'while -6 is roughly twice the bitrate. General '
                             'usage is to choose the highest CRF value that '
                             'still provides an acceptable quality. If the '
                             'output looks good, then try a higher value and '
                             'if it looks bad then choose a lower value. A '
                             'subjectively sane range is 18 to 28. (default: '
                             '%(default)s)',
                        metavar='23')

    parser.add_argument('-pts', action='store', default=1, type=float,
                        help='Presentation TimeStamp, pts < 1 to speedup the '
                             'video, pts > 1 to slow down the video. For '
                             'example, 0.5 will double the speed and cut the '
                             'video duration to half whereas 2 will slow the '
                             'video down and double its duration. (default: '
                             '%(default)s)',
                        metavar='1')

    parser.add_argument('-fps', action='store', default=30, type=float,
                        help='Frames Per Second of the output video. I you '
                             'speed a video up framesget dropped as ffmpeg '
                             'is trying to fit more frames in less time. '
                             'Try increasing the frame rate by the same '
                             'factor used for pts. If youslow a video down, '
                             'increasing the frame rate results in smoother '
                             'videofor some reason. (default: %(default)s)',
                        metavar='30')

    parser.add_argument('-q', action='store_true',
                        help='Suppress any output.')

    parser.add_argument('-i', action='store', default='./',
                        help='Path to where the sequential .png images are. '
                             '(default: %(default)s)',
                        metavar='./')

    parser.add_argument('outfile', action='store',
                        help='Name of the final .mp4 file.',)
    args = parser.parse_args(argv)

    png2mp4(args.outfile, args.i, args.crf, args.pts, args.fps, args.q)

if __name__ == "__main__":
    main()
