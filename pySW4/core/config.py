# -*- coding: utf-8 -*-
import os
import warnings
from obspy.core.util import AttribDict


def read_input_file(filename):
    """
    Parses an SW4 input file to a nested
    :class:`obspy.core.util.attribdict.AttribDict` / list /
    :class:`obspy.core.util.attribdict.AttribDict` structure.

    :type filename: str
    :param filename: Filename (potentially with relative/absolute path) of SW4
        config/input file.
    :rtype: :class:`obspy.core.util.attribdict.AttribDict`
    :returns: Parsed SW4 simulation input/config file.
    """
    with open(filename) as fh:
        lines = fh.readlines()
    config = AttribDict()
    for line in lines:
        # get rid of comments
        line = line.split("#")[0].strip()
        if not line:
            continue
        line = line.split()
        config_category = config.setdefault(line.pop(0), [])
        config_item = AttribDict()
        for item in line:
            key, value = item.split("=", 1)
            config_item[key] = _decode_string_value(value)
        config_category.append(config_item)
    return config


def _decode_string_value(string_item):
    """
    Converts string representations of int/float to the corresponding Python
    type.

    :type string_item: str
    :param string_item: Configuration value from SW4 input/config file in its
        string representation.
    :rtype: int or float or str
    :returns: Configuration value from SW4 input/config file as the correct
        Python type (bool values specified as `0` or `1` in SW4 config will
        still be of `int` type).
    """
    try:
        return int(string_item)
    except ValueError:
        pass
    try:
        return float(string_item)
    except ValueError:
        pass
    return string_item


def _parse_config_file_and_folder(config_file=None, folder=None):
    """
    Helper function to unify config location (or `None`) and output folder to
    work on.

    Use cases (in order of preference):

     * `config_file="/path/to/config", folder=None`:
       Config file is used for metadata and location of output folder
     * `config_file="/path/to/config", folder="/path/to/output"`:
       Config file is used for metadata, folder location is specified
       separately (make sure to not mismatch).
     * `config_file=None, folder="/path/to/output"`:
       Do not use metadata from config (station locations etc. will not show up
       in plots) and only use output files from specified location.
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
