# -*- coding: utf-8 -*-
import glob
import os
import re
import matplotlib.pyplot as plt
from seispy.image import read_SW4_image, image_files_to_movie


def create_all_plots(
        folder, source_time_function_type, frames_per_second=5,
        cmap=None):
    """
    Create all plots for an SW4 output folder.

    Currently always only uses first patch in each SW4 image file.
    """
    if not os.path.isdir(folder):
        msg = "Not a folder: '{}'".format(folder)
        raise ValueError(msg)
    all_files = glob.glob(os.path.join(folder, "*.sw4img"))
    if not all_files:
        msg = "No *.sw4img files in folder '{}'".format(folder)
        return Exception(msg)
    # build individual lists, one for each specific property
    grouped_files = {}
    for file_ in all_files:
        # e.g. shakemap.cycle=000.z=0.hmag.sw4img
        prefix, _, coordinate, type_ = \
            os.path.basename(file_).rsplit(".", 4)[:-1]
        grouped_files.setdefault((prefix, coordinate, type_), []).append(file_)
    for files in grouped_files.values():
        # create individual plots as .png
        for file_ in files:
            image = read_SW4_image(
                file_, source_time_function_type=source_time_function_type)
            outfile = file_.rsplit(".", 1)[0] + ".png"
            fig, _, _ = image.patches[0].plot(cmap=cmap)
            fig.savefig(outfile)
            plt.close(fig)
        # if several individual files in the group, also create a movie as .mp4
        if len(files) > 2:
            files = sorted(files)
            movie_filename = re.sub(
                r'([^.]*)\.cycle=[0-9]*\.(.*?)\.sw4img',
                r'\1.cycle=XXX.\2.mp4', files[0])
            image_files_to_movie(
                files, movie_filename, frames_per_second=frames_per_second,
                source_time_function_type=source_time_function_type,
                overwrite=True, cmap=cmap)

if __name__ == "__main__":
    create_all_plots(
        "/tmp/UH_01_simplemost", source_time_function_type="velocity")
