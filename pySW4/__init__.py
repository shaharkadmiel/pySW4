# -*- coding: utf-8 -*-
"""
pySW4: Python routines for interaction with SW4
===============================================
                 ______       ____ __
    ____  __  __/ ___/ |     / / // /
   / __ \/ / / /\__ \| | /| / / // /_
  / /_/ / /_/ /___/ /| |/ |/ /__  __/
 / .___/\__, //____/ |__/|__/  /_/
/_/    /____/

pySW4 is an open-source project dedicated to provide a Python framework
for working with numerical simulations of seismic-wave propagation with
SW4 in all phases of the task (preprocessing, post-processing and
runtime visualization).

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)
    Omry Volk (omryv@post.bgu.ac.il)
    Tobias Megies (megies@geophysik.uni-muenchen.de)

:license:
    GNU GENERAL PUBLIC LICENSE Version 3
    (http://www.gnu.org/copyleft/gpl.html)
    See ./LICENSE.txt
"""
from __future__ import absolute_import, print_function, division

__version__ = '0.2.2'

from .core.image import read_image
from .plotting.drape import drape_plot
from . import preprocess
from . import utils
