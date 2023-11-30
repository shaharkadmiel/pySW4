                         ______       ____ __
            ____  __  __/ ___/ |     / / // /
           / __ \/ / / /\__ \| | /| / / // /_
          / /_/ / /_/ /___/ /| |/ |/ /__  __/
         / .___/\__, //____/ |__/|__/  /_/
        /_/    /____/



pySW4: Python routines for interaction with SW4
===============================================

pySW4 is an open-source project dedicated to provide a Python framework for working with numerical simulations of seismic-wave propagation with [SW4](https://geodynamics.org/resources/sw4/about) in all phases of the task. Preprocessing, post-processing and runtime visualization. It also provides several other tools for handling GeoTIFF files such as ASTER-GDEM tiles and others, plotting shaded-relief DEM maps and draping data over them, creating movies from image sequences, etc.

The functionality is provided through 5 sub-packages which include pre- and post-processing routines including the [rfileIO](http://shaharkadmiel.github.io/pySW4/packages/pySW4.prep.rfileIO.html) library for interaction, reading and writing seismic models in the rfile format.

There are some useful utilities for geodesic projections of data and for reading and writing GeoTIFF files.

In the command line interface [scripts](http://shaharkadmiel.github.io/pySW4/packages/pySW4.cli.html), there are some quick and dirty plotting routines which can be run from the command line. It may be useful to run these scripts on the server-end while the computation is running in order to generate *pseudo-RunTime* visualization of the results.

See the full [API documentation](http://shaharkadmiel.github.io/pySW4/) page and [Tutorial examples](http://shaharkadmiel.github.io/pySW4/packages/tutorials.html).

Installation
------------
**conda**

Installing ``pySW4`` from the conda-forge channel can be achieved by
adding conda-forge to your channels with:

    $ conda config --add channels conda-forge

Once the conda-forge channel has been enabled, ``pySW4`` can be
installed with:

    $ conda install pysw4

It is possible to list all of the versions of ``pySW4`` available on
your platform with:

    $ conda search pysw4 --channel conda-forge
    
**pip**

You can install the repository directly from GitHub. Use this command to install from ``master``:

    $ pip install https://github.com/shaharkadmiel/pySW4/archive/master.zip

To get the latest release version do::

    $ pip install https://github.com/shaharkadmiel/pySW4/archive/v0.3.0.zip

Add the ``--no-deps`` to forgo dependency upgrade ot downgrade.

Quick and dirty plotting from the command line
----------------------------------------------

Plots and movies for a SW4 simulation run can be created using the
`pySW4-create-plots` command line utility, e.g.:

```bash
$ pySW4-create-plots -c /tmp/UH_01_simplemost.in -s UH1 -s UH2 -s UH3 \\
    --pre-filt 0.05,0.1,100,200 --filter type=lowpass,freq=10,corners=4 \\
    ~/data/stations/stationxml/*UH* ~/data/waveforms/*20100622214704*
```

![Wavefield](/images/shakemap.cycle=286.z=0.hmag.png)
![PGV map](/images/shakemap.cycle=477.z=0.hmax.png)
![Seismograms](/images/seismograms.png)
