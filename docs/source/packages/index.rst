=======================
pySW4 Library Reference
=======================

.. automodule:: pySW4
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

.. Introduction
.. ------------
.. The core package provides the main post-processing routines of the
.. pySW4 package. It has some scripts which can be run from the command
.. line for quick plotting. It may be usefull to run these scrips on
.. the server-end, even while the computation is running in order to
.. generate *pseudo-RunTime* visualization of the results.
.. See the :mod:`~pySW4.core.scripts` documentations for more
.. information.

The functionality is provided through the following sub-packages:
-----------------------------------------------------------------

.. autosummary::
   :toctree: .

    ~pySW4.cli
    ~pySW4.plotting
    ~pySW4.postp
    ~pySW4.prep
    ~pySW4.utils

Submodules
----------

.. autosummary::
   :toctree: .

    ~pySW4.headers
    ~pySW4.sw4_metadata
