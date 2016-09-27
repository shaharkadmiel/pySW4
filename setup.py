import inspect
import os
import re
from setuptools import setup

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'matplotlib',
    'obspy',
    'pillow',
    'gdal',
    ]

SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))
ENTRY_POINTS = {
    'console_scripts': [
        'pySW4-create-plots = pySW4.core.scripts.create_all_plots:main',
        'png2mp4 = pySW4.core.scripts.png2mp4:main']}


def find_packages():
    """
    Simple function to find all modules under the current folder.
    """
    modules = []
    for dirpath, _, filenames in os.walk(
            os.path.join(SETUP_DIRECTORY, "pySW4")):
        if "__init__.py" in filenames:
            modules.append(os.path.relpath(dirpath, SETUP_DIRECTORY))
    return [_i.replace(os.sep, ".") for _i in modules]

# get the package version from from the main __init__ file.
version_regex_pattern = r"__version__ += +(['\"])([^\1]+)\1"
for line in open('pySW4/__init__.py'):
    if '__version__' in line:
        package_version = re.match(version_regex_pattern, line).group(2)

setup(
    name="pySW4",
    version=package_version,
    description="Python routines for interaction with SW4",
    author="Shahar Shani-Kadmiel, Omry Volk, Tobias Megies",
    author_email="kadmiel@post.bgu.ac.il",
    url="https://github.com/shaharkadmiel/pySW4",
    download_url="https://github.com/shaharkadmiel/pySW4.git",
    install_requires=INSTALL_REQUIRES,
    keywords=["pySW4", "seismology", "SW4"],
    packages=find_packages(),
    entry_points=ENTRY_POINTS,
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description='pySW4 is an open-source project dedicated to '
                     'provide a Python framework for working with '
                     'numerical simulations of seismic-wave propagation '
                     'with SW4 in all phases of the task (preprocessing, '
                     'post-processing and runtime visualization).'
    )
