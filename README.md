                         ______       ____ __
            ____  __  __/ ___/ |     / / // /
           / __ \/ / / /\__ \| | /| / / // /_
          / /_/ / /_/ /___/ /| |/ |/ /__  __/
         / .___/\__, //____/ |__/|__/  /_/
        /_/    /____/



pySW4: Python routines for interaction with SW4
===============================================

pySW4 is an open-source project dedicated to provide a Python framework for
working with numerical simulations of seismic-wave propagation with SW4 in all
phases of the task (preprocessing, post-processing and runtime visualization).

Installation
------------

```bash
$ pip install https://github.com/shaharkadmiel/pySW4/archive/master.zip
```

Updating a previoius install:

```bash
$ pip install -U https://github.com/shaharkadmiel/pySW4/archive/master.zip
```

or without updating dependencies:

```bash
$ pip install -U --no-deps https://github.com/shaharkadmiel/pySW4/archive/master.zip
```

or a specific branch:

```bash
$ pip install -U --no-deps https://github.com/shaharkadmiel/pySW4/archive/0.3.0.zip
```

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
