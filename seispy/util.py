# -*- coding: utf-8 -*-
import glob
import os
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import obspy
from seispy.header import SW4_SOURCE_TIME_FUNCTION_TYPE
from seispy.image import read_SW4_image, image_files_to_movie
from seispy.config import read_input_file


def _parse_config_file_and_folder(config_file=None, folder=None):
    """
    """
    if config_file is None and folder is None:
        msg = ("At least one of `config_file` or `folder` has to be "
               "specified.")
        raise ValueError(msg)

    if config_file:
        config_folder = os.path.dirname(os.path.abspath(config_file))
        config = read_input_file(config_file)
    else:
        config = None

    if config:
        folder_ = os.path.join(config_folder, config.fileio[0].path)
        if folder and os.path.abspath(folder) != folder_:
            msg = ("Both `config` and `folder` option specified. Overriding "
                   "folder found in config file ({}) with user specified "
                   "folder ({}).").format(folder_, folder)
            warnings.warn(msg)
        else:
            folder = folder_
    return config, folder


def create_image_plots(
        config_file=None, folder=None, source_time_function_type=None,
        frames_per_second=5, cmap=None, movies=True):
    """
    Create all image plots/movies for a SW4 run.

    Currently always only uses first patch in each SW4 image file.
    If the path/filename of the SW4 input file is provided, additional
    information is included in the plots (e.g. receiver/source location,
    automatic determination of source time function type, ..)
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
            image = read_SW4_image(
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
                image_files_to_movie(
                    files, movie_filename, frames_per_second=frames_per_second,
                    source_time_function_type=source_time_function_type,
                    overwrite=True, cmap=cmap, config=config)


# SW4 uses a right handed coordinate system:
#   - X == Northing
#   - Y == Easting
#   - Z == Vertical (inverted!)
#
#    _ X
#    /`
#   /
#  /_____\  y
#  |     /
#  |
#  V
#   Z
def create_seismogram_plots(
        config_file, folder, stream_observed, inventory, water_level, pre_filt,
        filter_kwargs=None, channel_map={"-Vz": "Z", "Vx": "N", "Vy": "E"},
        used_stations=None, synthetic_starttime=None):
    """
    """
    config, folder = _parse_config_file_and_folder(config_file, folder)

    stf_type = SW4_SOURCE_TIME_FUNCTION_TYPE[config.source[0].type]
    if stf_type == 0:
        evalresp_output = "DISP"
        unit_label = "m"
    elif stf_type == 1:
        evalresp_output = "VEL"
        unit_label = "m/s"
    else:
        raise NotImplementedError()

    st_synth = obspy.read(os.path.join(folder, "*.?v"))
    st_real = stream_observed
    if used_stations is not None:
        st_synth.traces = [tr for tr in st_synth
                           if tr.stats.station in used_stations]
    stations = set([tr.stats.station for tr in st_synth])
    st_real.traces = [tr for tr in st_real
                      if tr.stats.station in stations]
    for tr in st_synth:
        # SW4 vertical channel is positive in coordinate direction, which
        # is depth positive down. So we have to invert it to get the normal
        # seismometer vertical up trace.
        if tr.stats.channel == "Vz":
            tr.stats.channel = "-Vz"
            tr.data *= -1
    t_min = min([tr.stats.starttime for tr in st_synth])
    t_max = min([tr.stats.endtime for tr in st_synth])
    st_real.attach_response(inventory)
    st_real.remove_response(
        output=evalresp_output, water_level=water_level, pre_filt=pre_filt)
    if filter_kwargs:
        st_real.filter(**filter_kwargs)
    st_real.trim(t_min, t_max)

    outfile = os.path.join(folder, "seismograms.png")
    _plot_seismograms(st_synth, st_real, channel_map, unit_label, outfile)
    for station in stations:
        outfile = os.path.join(folder, "seismograms.{}.png".format(station))
        st_synth_ = st_synth.select(station=station)
        st_real_ = st_real.select(station=station)
        _plot_seismograms(
            st_synth_, st_real_, channel_map, unit_label, outfile,
            figsize=(10, 8))


def _plot_seismograms(
        st_synth_, st_real_, channel_map, unit_label, outfile, figsize=None):
    """
    """
    if figsize is None:
        figsize = (10, len(st_synth_))

    fig = plt.figure(figsize=figsize)
    st_synth_.plot(fig=fig)

    # print("Real data:")
    # print(st_real)
    # print(st_real.max())
    # print("Synthetic data:")
    # print(st_synth)
    # print(st_synth.max())

    for ax in fig.axes:
        id = ax.texts[0].get_text()
        _, sta, _, cha = id.split(".")
        real_component = channel_map[cha]
        # find appropriate synthetic trace
        for tr_real in st_real_:
            if tr_real.stats.station != sta:
                continue
            # SW4 synthetics channel codes (for velocity traces) are "V[xyz]"
            if tr_real.stats.channel[-1] != real_component:
                continue
            break
        else:
            continue
        ax.text(0.95, 0.9, tr_real.id, ha="right", va="top", color="r",
                transform=ax.transAxes)
        t = date2num([tr_real.stats.starttime + t_ for
                      t_ in tr_real.times()])
        ax.plot(t, tr_real.data, "r-")
        ax.set_ylabel(unit_label)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, hspace=0.0, wspace=0.0)
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    create_image_plots(
        # config_file="/tmp/UH_01_simplemost.in",
        # folder="/tmp/UH_01_simplemost33")
        config_file="/tmp/UH_01_simplemost.in")
