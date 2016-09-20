# -*- coding: utf-8 -*-
"""
Plotting routines for SW4 images of Maps or Cross-Sections

.. module:: image

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

import glob
import os
import time
import re
import subprocess as sub
from warnings import warn
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
try:
    from obspy.imaging.cm import obspy_divergent as cmap_divergent
    from obspy.imaging.cm import obspy_divergent_r as cmap_divergent_r
    from obspy.imaging.cm import obspy_sequential as cmap_sequential
    from obspy.imaging.cm import obspy_sequential_r as cmap_sequential_r
except ImportError:
    cmap_divergent = None
    cmap_divergent_r = None
    cmap_sequential = None
    cmap_sequential_r = None

from ..postp import read_image
from ..postp.input import _parse_input_file_and_folder


def png2mp4(outfile, inpath='./', crf=23, pts=1, fps=30, verbose=True):
    """
    Python module for creating mp4 annimations from a set of
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


def image_files_to_movie(
        input_filenames, output_filename, input_file=None,
        stf=None, patch_number=0,
        frames_per_second=5, overwrite=False, global_colorlimits=True,
        debug=False, **plot_kwargs):
    """
    Convert SW4 images to an mp4 movie using command line ffmpeg.

    Parameters
    ----------
    input_filenames : str or list
        Wildcarded filename pattern or list of individual filenames.

    output_filename : str
        Output movie filename ('.mp4' extension will be appended if not
        already present).
    """
    if not output_filename.endswith(".mp4"):
        output_filename += ".mp4"
    if os.path.exists(output_filename):
        if overwrite:
            os.remove(output_filename)
    if os.path.exists(output_filename):
        msg = ("Output path '{}' exists.").format(output_filename)
        raise IOError(msg)

    if isinstance(input_filenames, str):
        files = sorted(glob.glob(input_filenames))
    else:
        files = input_filenames

    # parse all files to determine global value range extrema
    # before plotting
    if global_colorlimits:
        global_min = np.inf
        global_max = -np.inf
        for file_ in files:
            image = read_image(
                file_, input_file=input_file,
                stf=stf)
            patch = image.patches[patch_number]
            global_min = min(global_min, patch.min)
            global_max = max(global_max, patch.max)
        if image.is_divergent:
            abs_max = max(abs(global_min), abs(global_max))
            plot_kwargs["vmin"] = -abs_max
            plot_kwargs["vmax"] = abs_max
        else:
            plot_kwargs["vmin"] = global_min
            plot_kwargs["vmax"] = global_max
        if global_min == np.inf and global_max == -np.inf:
            msg = ("Invalid global data limits for files '{}'").format(
                input_filenames)
            raise ValueError(msg)

    cmdstring = (
        'ffmpeg', '-loglevel', 'fatal', '-r', '%d' % frames_per_second,
        '-f', 'image2pipe', '-vcodec', 'png', '-i', 'pipe:',
        '-vcodec', 'libx264', '-pass', '1', '-vb', '6M', '-pix_fmt', 'yuv420p',
        output_filename)

    string_io = StringIO()
    backend = plt.get_backend()
    try:
        plt.switch_backend('AGG')
        # plot all images and pipe the pngs to ffmpeg
        for file_ in files:
            image = read_image(
                file_, stf=stf,
                input_file=input_file)
            patch = image.patches[patch_number]
            fig, _, _ = patch.plot(**plot_kwargs)
            fig.tight_layout()
            fig.savefig(string_io, format='png')
            plt.close(fig)
        string_io.seek(0)
        png_data = string_io.read()
        string_io.close()
        sub = sub.Popen(cmdstring, stdin=sub.PIPE,
                        stdout=sub.PIPE, stderr=sub.PIPE)
        stdout, stderr = sub.communicate(png_data)
        if debug:
            print("###### ffmpeg stdout")
            print(stdout)
            print("###### ffmpeg stderr")
            print(stderr)
        sub.wait()
        for ffmpeg_tmp_file in ("ffmpeg2pass-0.log",
                                "ffmpeg2pass-0.log.mbtree"):
            if os.path.exists(ffmpeg_tmp_file):
                os.remove(ffmpeg_tmp_file)
    finally:
        plt.switch_backend(backend)


def create_image_plots(
        input_file, folder=None, stf=None,
        frames_per_second=5, cmap=None, movies=True):
    """
    Create all image plots/movies for a SW4 run.

    Currently always only uses first patch in each SW4 image file.
    If the path/filename of the SW4 input file is provided, additional
    information is included in the plots (e.g. receiver/source location,
    automatic determination of source time function type, etc...)

    Parameters
    ----------
    input_file : str
        Filename (potentially with absolute/relative path) of SW4 input
        file used to control the simulation. Use ``None`` to work on
        folder with SW4 output without using metadata from input file.

    folder : str
        Folder with SW4 output files or ``None`` if output folder
        location can be used from input file. Only needed when no
        input file is specified or if output folder was moved to a
        different location after the simulation.

    stf : str
        ``'displacement'`` or ``'velocity'``.

    frames_per_second : int or float
        Image frames to show per second in output videos.

    cmap : str or :class:`~matplotlib.colors.Colormap`
        Matplotlib colormap or colormap string understood by matplotlib.

    movies : bool
        Whether to produce movies from image files present at different
        cycles of the simulation. Needs `ffmpeg`_ to be installed and
        callable on command line.

    .. _ffmpeg:
       https://ffmpeg.org/
    """
    input, folder = _parse_input_file_and_folder(input_file, folder)

    if stf is None and input is None:
        msg = ("No input file specified (option `input_file`) "
               "and source time function type not specified explicitely "
               "(option `stf`).")
        raise ValueError(msg)

    if not os.path.isdir(folder):
        msg = "Not a folder: '{}'".format(folder)
        raise ValueError(msg)

    all_files = glob(os.path.join(folder, "*.sw4img"))
    if not all_files:
        msg = "No *.sw4img files in folder '{}'".format(folder)
        return Exception(msg)

    # build individual lists, one for each specific property
    grouped_files = {}
    for file_ in all_files:
        # e.g. shakemap.cycle=000.z=0.hmag.sw4img
        prefix, _, coordinate, type_ = \
            os.path.basename(file_).rsplit(".", 4)[:-1]
        grouped_files.setdefault((prefix, coordinate, type_), []).append(file_)
    for files in grouped_files.values():
        # create individual plots as .png
        for file_ in files:
            image = read_image(
                file_, stf=stf,
                input_file=input_file)
            outfile = file_.rsplit(".", 1)[0] + ".png"
            fig, _, _ = image.patches[0].plot(cmap=cmap)
            fig.savefig(outfile)
            plt.close(fig)
        # if several individual files in the group, also create a movie as .mp4
        if movies:
            if len(files) > 2:
                files = sorted(files)
                movie_filename = re.sub(
                    r'([^.]*)\.cycle=[0-9]*\.(.*?)\.sw4img',
                    r'\1.cycle=XXX.\2.mp4', files[0])
                try:
                    image_files_to_movie(
                        files, movie_filename,
                        frames_per_second=frames_per_second,
                        stf=stf,
                        overwrite=True, cmap=cmap, input_file=input_file)
                except Exception as e:
                    msg = ("Failed to create a movie: {}").format(str(e))
                    warn(msg)
