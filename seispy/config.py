# -*- coding: utf-8 -*-
from obspy.core.util import AttribDict


def read_input_file(filename):
    """
    Parses an SW4 input file to a nested
    :class:`obspy.core.util.attribdict.AttribDict` / list /
    :class:`obspy.core.util.attribdict.AttribDict` structure.
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
