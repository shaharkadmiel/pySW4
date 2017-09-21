.. _index:

===================
pySW4 documentation
===================

Introduction
------------
pySW4 is an open-source project dedicated to provide a Python framewor
for working with numerical simulations of seismic-wave propagation with
SW4 in all phases of the task (preprocessing, post-processing and
runtime visualization). The functionality is provided through 5
sub-packages which include pre- and post-processing routines including
the :mod:`~pySW4.prep.rfileIO` library for interaction, reading and
writing seismic models in the rfile format.

There are some usefull utilities for geodesic projections of data and
for reading and writing GeoTIFF files.

In the command line interface scrips, there are some quick
and dirty plotting routines which can be run from the command
line. It may be usefull to run these scrips on the server-end while the
computation is running in order to generate *pseudo-RunTime*
visualization of the results.

Installation
------------
**conda**

Installing ``pySW4`` from the conda-forge channel can be achieved by
adding conda-forge to your channels with::

    conda config --add channels conda-forge

Once the conda-forge channel has been enabled, ``pySW4`` can be
installed with::

    conda install pysw4

It is possible to list all of the versions of ``pySW4`` available on
your platform with::

    conda search pysw4 --channel conda-forge

**pip**

You can install the repository directly from GitHub. Use this command to install from ``master``::

    pip install https://github.com/shaharkadmiel/pySW4/archive/master.zip

To get the latest release version do::

    pip install https://github.com/shaharkadmiel/pySW4/archive/v0.3.0.zip

Add the ``--no-deps`` to forgo dependency upgrade ot downgrade.

Documentation and tutorials
---------------------------
.. toctree::
   :maxdepth: 1

   packages/index
   packages/tutorials

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`