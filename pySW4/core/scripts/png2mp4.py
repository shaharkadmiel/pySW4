# -*- coding: utf-8 -*-
"""
Convert a sequencial set of .png images to a .mp4 movie

.. module:: png2mp4

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

import sys
import os
import time
import argparse
import subprocess as sub


def png2mp4(outfile, inpath='./', crf=23, pts=1, fps=30, verbose=True):
    """ Python module for creating mp4 annimations from a set of
    sequential png images.

    Parameters
    ----------

    outfile : str
        Name of the final .mp4 file.

    inpath : str
        Path to where the sequential .png images are. Default is `'./'`.

    crf : int
        Constant Rate Factor, ranges from 0 to 51: A lower value is a
        higher quality and vise versa. The range is exponential, so
        increasing the CRF value +6 is roughly half the bitrate while -6
        is roughly twice the bitrate. General usage is to choose the
        highest CRF value that still provides an acceptable quality. If
        the output looks good, then try a higher value and if it looks
        bad then choose a lower value. A subjectively sane range is 18
        to 28. Default is 23.

    pts : int
        Presentation TimeStamp, pts < 1 to speedup the video, pts > 1 to
        slow down the video. For example, 0.5 will double the speed and
        cut the video duration in half whereas 2 will slow the video
        down and double its duration. Default is 1.

    fps : int
        Frames Per Second of the output video. If you speed a video up
        frames get dropped as ffmpeg is trying to fit more frames in
        less time. Try increasing the frame rate by the same factor used
        for pts. If you slow a video down, increasing the frame rate
        results in smoother video for some reason. Default is 30.

    verbose : bool
        Print information about the process if True. Set to False to
        suppress any output.
    """

    inpath = os.path.join(inpath, '*.png')
    if os.path.splitext(outfile)[-1] not in ['.mp4', '.MP4']:
        outfile += '.mp4'

    if verbose:
        print('*** converting sequencial png files to mp4...\n')
        sys.stdout.flush()

    t = time.time()
    command = ("ffmpeg -y -pattern_type glob -i {} "
               "-vcodec libx264 -crf {} -pass 1 -vb 6M "
               "-pix_fmt yuv420p "
               "-vf scale=trunc(iw/2)*2:trunc(ih/2)*2,setpts={}*PTS "
               "-r {} -an {}").fotmat(inpath, crf, pts, fps, outfile)

    command = command.split()

    if verbose:
        print('***\ncalling {} with the following arguments:\n'
              .format(command[0]))
        for item in command[1:]:
            print(item, end="")
        print('\n***\n')
        sys.stdout.flush()

    time.sleep(1)

    p = sub.Popen(command,
                  stdin=sub.PIPE,
                  stdout=sub.PIPE,
                  stderr=sub.PIPE)
    p.wait()

    if verbose:
        print('\n******\nconvertion took {} seconds'.format(time.time() - t))


def main(argv=None):

    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: {}\n'.format(message))
            self.print_help()
            sys.exit(2)

    parser = DefaultHelpParser(description='Convert a sequencial set of '
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
