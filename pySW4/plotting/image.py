# -*- coding: utf-8 -*-
"""
Module to handle SW4 images of Maps or Cross-Sections

By: Omri Volk, Shahar Shani-Kadmiel and Tobias Megies, 2015-2016,
    kadmiel@post.bgu.ac.il
"""
from __future__ import absolute_import, print_function, division

import glob
import os
import re
import subprocess
import warnings
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

from ..core import read_image
from ..core.config import _parse_config_file_and_folder
from .util import set_matplotlib_rc_params


set_matplotlib_rc_params()


def image_files_to_movie(
        input_files, output_filename, config=None,
        source_time_function_type=None, patch_number=0,
        frames_per_second=5, overwrite=False, global_colorlimits=True,
        debug=False, **plot_kwargs):
    """
    Convert SW4 images to an mp4 movie using command line ffmpeg.

    :type input_files: str or list
    :param input_files: Wildcarded filename pattern or list of individual
        filenames.
    :type output_filename: str
    :param output_filename: Output movie filename ('.mp4' extension will be
        appended if not already present).
    """
    if not output_filename.endswith(".mp4"):
        output_filename += ".mp4"
    if os.path.exists(output_filename):
        if overwrite:
            os.remove(output_filename)
    if os.path.exists(output_filename):
        msg = ("Output path '{}' exists.").format(output_filename)
        raise IOError(msg)

    if isinstance(input_files, str):
        files = sorted(glob.glob(input_files))
    else:
        files = input_files

    # parse all files to determine global value range extrema before doing
    # the plotting
    if global_colorlimits:
        global_min = np.inf
        global_max = -np.inf
        for file_ in files:
            image = read_image(
                file_, config=config,
                source_time_function_type=source_time_function_type)
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
                input_files)
            raise ValueError(msg)

    cmdstring = (
        'ffmpeg', '-loglevel', 'fatal',  '-r', '%d' % frames_per_second,
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
                file_, source_time_function_type=source_time_function_type,
                config=config)
            patch = image.patches[patch_number]
            fig, _, _ = patch.plot(**plot_kwargs)
            fig.tight_layout()
            fig.savefig(string_io, format='png')
            plt.close(fig)
        string_io.seek(0)
        png_data = string_io.read()
        string_io.close()
        sub = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        config_file, folder=None, source_time_function_type=None,
        frames_per_second=5, cmap=None, movies=True):
    """
    Create all image plots/movies for a SW4 run.

    Currently always only uses first patch in each SW4 image file.
    If the path/filename of the SW4 input file is provided, additional
    information is included in the plots (e.g. receiver/source location,
    automatic determination of source time function type, ..)

    :type config_file: str
    :param config_file: Filename (potentially with absolute/relative path) of
        SW4 input/config file used to control the simulation. Use `None` to
        work on folder with SW4 output without using metadata from config.
    :type folder: str
    :param folder: Folder with SW4 output files or `None` if output folder
        location can be used from config file. Only needed when no config file
        is specified or if output folder was moved to a different location
        after the simulation.
    :type source_time_function_type: str
    :param source_time_function_type: `displacement` or `velocity`.
    :type frames_per_second: float
    :param frames_per_second: Image frames to show per second in output videos.
    :type cmap: str or :class:`matplotlib.colors.Colormap`
    :param cmap: Matplotlib colormap or colormap string understood by
        matplotlib.
    :type movies: bool
    :param movies: Whether to produce movies from image files present at
        different cycles of the simulation. Needs `ffmpeg` to be installed and
        callable on command line.
    """
    config, folder = _parse_config_file_and_folder(config_file, folder)

    if source_time_function_type is None and config is None:
        msg = ("No input configuration file specified (option `config_file`) "
               "and source time function type not specified explicitely "
               "(option `source_time_function_type`).")
        ValueError(msg)

    if not os.path.isdir(folder):
        msg = "Not a folder: '{}'".format(folder)
        raise ValueError(msg)

    all_files = glob.glob(os.path.join(folder, "*.sw4img"))
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
                file_, source_time_function_type=source_time_function_type,
                config=config)
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
                        source_time_function_type=source_time_function_type,
                        overwrite=True, cmap=cmap, config=config)
                except Exception as e:
                    msg = ("Failed to create a movie: {}").format(str(e))
                    warnings.warn(msg)
