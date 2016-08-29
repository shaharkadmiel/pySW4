# -*- coding: utf-8 -*-
"""
Create all plots for an SW4 simulation run.
"""
from __future__ import absolute_import, print_function, division

from argparse import ArgumentParser
import warnings

from obspy import read, Stream, read_inventory
from obspy.core.inventory import Inventory

from pySW4 import __version__
from pySW4.core.config import _decode_string_value
from pySW4.plotting import create_image_plots, create_seismogram_plots


def main(argv=None):
    parser = ArgumentParser(prog='pySW4-create-plots',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)

    parser.add_argument('-c', '--config', dest='config_file', default=None,
                        type=str,
                        help='SW4 input/config file of simulation run.')

    parser.add_argument('-f', '--folder', dest='folder', default=None,
                        type=str,
                        help='Folder with SW4 simulation run output. '
                             'Can be omitted if a config file is specified '
                             'and output directory was not moved after the '
                             'simulation.')

    parser.add_argument('--no-movies', dest='no_movies', action="store_true",
                        default=False, help='Omits creation of mp4 movies.')

    parser.add_argument('-s', '--use-station', dest="used_stations",
                        action="append", default=None, type=str,
                        help='Station codes of stations to consider for '
                             'seismogram plots. Can be specified multiple '
                             'times. If not specified, all stations which '
                             'have synthetic data output are used.')

    parser.add_argument('--water-level', dest="water_level", default=10,
                        type=float,
                        help='Water level used in instrument response '
                             'removal of real/observed data (see '
                             'obspy.core.trace.Trace.remove_response for '
                             'details).')

    parser.add_argument('--pre-filt', dest="pre_filt",
                        default=[0.05, 0.1, 100, 200],
                        help='Frequency domain pre-filter used in instrument '
                             'response removal of real/observed data (see '
                             'obspy.core.trace.Trace.remove_response for '
                             'details). Four comma-separated float values '
                             '(e.g. "0.1,0.5,40,80").')

    parser.add_argument('--filter', dest="filter_kwargs", default="",
                        type=str,
                        help='Filter to be applied on real/observed data '
                             'after instrument response removal. '
                             'Comma-separated kwargs as understood by '
                             'obspy.core.trace.Trace.filter (e.g. '
                             '"type=bandpass,freqmin=1,freqmax=5").')

    parser.add_argument('real_data_files', nargs='+',
                        help='Files with observed/real data (waveform files '
                             'readable by obspy.read (e.g. MSEED, SAC, ..) '
                             'and station metadata files readable by '
                             'obspy.read_inventory (e.g. StationXML).')

    args = parser.parse_args(argv)

    if args.filter_kwargs:
        args.filter_kwargs = [
            item.split("=") for item in args.filter_kwargs.split(",")]
        args.filter_kwargs = dict([
            (k, _decode_string_value(v))
            for k, v in args.filter_kwargs])
    if isinstance(args.pre_filt, str):
        args.pre_filt = map(float, args.pre_filt.split(","))

    st = Stream()
    inv = Inventory(networks=[], source="")
    msg = ("Could not read real/observed data file using obspy.read or "
           "obspy.read_inventory: '{}'")
    for f in args.real_data_files:
        try:
            inv += read_inventory(f)
        except:
            pass
        else:
            continue
        try:
            st += read(f)
        except:
            pass
        else:
            continue
        warnings.warn(msg.format(f))

    create_image_plots(config_file=args.config_file,
                       movies=not args.no_movies)
    create_seismogram_plots(config_file=args.config_file, folder=args.folder,
                            stream_observed=st, inventory=inv,
                            used_stations=args.used_stations,
                            water_level=args.water_level,
                            pre_filt=args.pre_filt,
                            filter_kwargs=args.filter_kwargs)


if __name__ == "__main__":
    main()
