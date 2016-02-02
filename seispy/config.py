# -*- coding: utf-8 -*-
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
        line = line.strip().split()
        if not line or line[0].startswith("#"):
            continue
        config_category = config.setdefault(line.pop(0), [])
        config_item = AttribDict()
        for item in line:
            key, value = item.split("=")
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
