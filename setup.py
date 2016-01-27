import inspect
import os
from setuptools import setup

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'matplotlib',
    'obspy',
    'PIL',
    'gdal',
    ]
SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))


def find_packages():
    """
    Simple function to find all modules under the current folder.
    """
    modules = []
    for dirpath, _, filenames in os.walk(
            os.path.join(SETUP_DIRECTORY, "seispy")):
        if "__init__.py" in filenames:
            modules.append(os.path.relpath(dirpath, SETUP_DIRECTORY))
    return [_i.replace(os.sep, ".") for _i in modules]


setup(
    name="seispy",
    version="0.1.0",
    description="A Python Toolbox for processing Seismic-wave propagation "
                "simulations",
    author="Shahar Shani-Kadmiel, Omry Volk, Tobias Megies",
    author_email="kadmiel@post.bgu.ac.il",
    url="https://github.com/shaharkadmiel/seispy",
    download_url="https://github.com/shaharkadmiel/seispy.git",
    install_requires=INSTALL_REQUIRES,
    keywords=["seispy", "seismology", "SW4", "WPP"],
    packages=find_packages(),
    entry_points={},
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Library or " +
        "General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description="""\
SeisPy is an open-source project dedicated to provide a Python framework for
processing numerical simulations of seismic-wave propagation in all phases of
the task (preprocessing, post-processing and runtime visualization).
"""
)
