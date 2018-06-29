# -*- coding: utf-8 -*-
"""
pySW4: Python routines for interaction with SW4
===============================================

**pySW4** is an open-source project dedicated to provide a Python
framework for working with numerical simulations of seismic-wave
propagation with SW4 in all phases of the task (preprocessing,
post-processing and runtime visualization).

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

__version__ = '0.3.5'

from . import utils
from . import plotting
from .postp import read_image, read_stations

from . import prep

from .sw4_metadata import Inputfile, Outputfile, read_metadata

from .plotting.utils import set_matplotlib_rc_params
set_matplotlib_rc_params()
