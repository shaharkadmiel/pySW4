# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Convenience imports for pySW4
#   Author: Shahar Shani-Kadmiel
#			      kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------
"""

                 ______       ____ __
    ____  __  __/ ___/ |     / / // /
   / __ \/ / / /\__ \| | /| / / // /_
  / /_/ / /_/ /___/ /| |/ |/ /__  __/
 / .___/\__, //____/ |__/|__/  /_/
/_/    /____/



pySW4: A Python Toolbox for processing Seismic-wave propagation simulations
============================================================================

pySW4 is an open-source project dedicated to provide a Python framework for
processing numerical simulations of seismic-wave propagation in all phases of
the task (preprocessing, post-processing and runtime visualization).

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)
    and
    Omry Volk (enter email here)
:license:
    GNU GENERAL PUBLIC LICENSE Version 3
    (http://www.gnu.org/copyleft/gpl.html)
    See ./LICENSE.txt
"""

from .fourier_spectrum import fourier_spectrum
from .psde import psde
from .geo import *
from .utils import *
