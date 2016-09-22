.. _index:

============
Core package
============

.. currentmodule:: pySW4.core
.. automodule:: pySW4.core

    Introduction
    ------------
    The core package provides the main post-processing routines of the
    pySW4 package. It has some scripts which can be run from the command
    line for quick plotting. It may be usefull to run these scrips on
    the server-end, even while the computation is running in order to
    generate *pseudo-RunTime* visualization of the results.
    See the :mod:`~pySW4.core.scripts` documentations for more
    information.

    Classes and Functions
    ---------------------
    .. autosummary::
        :toctree: .
        :nosignatures:

        ~pySW4.core.image.Image
        ~pySW4.core.image.Patch
        ~pySW4.core.image.read_image


    Scripts
    -------
    .. autosummary::
        :toctree: .
        :nosignatures:

        ~pySW4.core.scripts


    Modules
    -------
    .. autosummary::
        :toctree: .
        :nosignatures:

        image
        config
        header