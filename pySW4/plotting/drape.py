# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Draping data over shaded relief
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------
from __future__ import absolute_import, print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageChops, ImageEnhance

def drape_plot(relief, data, extent, intensity=None, data_mask=None,
               data_clipfactor='max', vmin='min', bg_brightness=1, fg_brightness=1,
               rcmap='Greys', icmap='Greys', dcmap='hot_r',
               origin='upper', ax=None, colorbar=True):

    """Drape `data` over `relief` or hill-shaded relief.
    If `intensity` is supplied, `relief` is first draped over `intensity`
    and then `data` is draped over both. Otherwise, `data` is draped
    over `relief` only.

    If supplied, intensity and relief arrays **MUST** be the same shape.

    Params :
    --------

    relief : a 2d array, usually of elevation

    data : a 2d array with the data to be draped

    extent : a list or tuple of the extent of the domain plotted.
        (xmin,xmax,ymin,ymax) or (w,e,s,n)

    intensity : a hill-shade intensities array, same shape as the relief
        array. If only a shaded relief is needed, pass the hill-shade
        intensity array as `data` and leave `intensity` as None.
        This will cause `relief` to be draped over `intensity`.

    data_mask : a 2d boolean array, same shape as data used to mask
        the `data` before draping is done.

    data_clipfactor : used to clip the coloring of the data at the set
        value, similar to vmax. Default is 'max' which shows all the data.
        If `data_clipfactor` is `int` or `float`, the clipping value is
        set to `data_clipfactor`. Finally, if a string is passed, it
        is casted to float and used an `rms` multiplier. For instance,
        if `data_clipfactor='3'`, clipping is done at 3.0*rms of the data.

    vmin : used as data minimum clipping. Default is `0`. If set to None or
        False, vmin is set to -vmax so that the coloring of the data is
        symmetric arround 0.

    bg_brightness and fg_brightness : control the brightness of the
        background (relief or shaded relief) and foreground (data).

    rcmap, icmap and dcmap : relief, intensity and data colormap

    origin : places the origin of the bg and fg at the 'upper' (default)
        or 'lower' left corner of the plot.

    ax : an axes instance to which to plot to. If None, a figure and axes
        are created and their instances are returned for further manipulation.

    colorbar : by default, a colorbar is drawn with the plot. If `colorbar`
        is a string, it is used for the label of the colorbar. Otherwise,
        the colorbar can be omitted by setting to False. colorbar instance
        is returned.

    ** Note **
    All grids must have the same extent and origin!

    """

    if not ax:
        fig, ax = plt.subplots()
        ax.axis(extent)

    # relief
    imR = ax.imshow(relief, extent=extent, cmap=rcmap, origin=origin)
    rgbR = imR.to_rgba(relief,bytes=True)#[:,:,:3]
    imR.remove()
    im1 = Image.fromarray(rgbR, mode='RGBA')

    # data
    if data_clipfactor is 'max':
        clip = np.nanmax(data)
    elif type(data_clipfactor) in [float, int]:
        clip = data_clipfactor
    else:
        clip = float(data_clipfactor)*np.nanstd(data)

    if vmin is 'min':
        vmin = np.nanmin(data)
    elif type(vmin) in [int,float]:
        pass
    elif vmin in [None, False]:
        vmin = -clip

    vmax = clip

    if vmin > data.min() and vmax < data.max():
        extend = 'both'
    elif vmin > data.min():
        extend = 'min'
    elif vmax < data.max():
        extend = 'max'
    else:
        extend = 'neither'

    imD = ax.imshow(data, extent=extent, cmap=dcmap, vmin=vmin, vmax=vmax, origin=origin)
    rgbD = imD.to_rgba(data,bytes=True)#[:,:,:3]
    imD.remove()
    im3 = Image.fromarray(rgbD, mode='RGBA')
    if data_mask is not None:
        im3 = maskimage(im3,data_mask)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(imD, cax=cax, extend=extend)
        if type(colorbar) is str:
            cb.set_label(colorbar)
        cb.solids.set_edgecolor("face")
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((-1,4))
        cb.update_ticks()

    # intensity
    if intensity is not None:
        imI = ax.imshow(intensity, extent=extent, cmap=icmap, origin=origin)
        rgbI = imI.to_rgba(intensity,bytes=True)#[:,:,:3]
        imI.remove()
        im2 = Image.fromarray(rgbI, mode='RGBA')

        # drape im1 on im2 (relief on intensity)
        imBG = ImageChops.multiply(im1,im2)
    else:
        imBG = im1

    # drape im3 (data) on imBG (data on background) after making sure
    # they are the same size. If they are not the same size,
    # supersample the low-res. image to the res. of the hi-res. image.
    if im3.size != imBG.size:
        if im3.size < imBG.size:
            im3 = im3.resize(imBG.size,2)
        elif im3.size > imBG.size:
            imBG = imBG.resize(im3.size,2)

    # enhance background image
    enhancerBG = ImageEnhance.Brightness(imBG)
    imBG = enhancerBG.enhance(bg_brightness)

    # drape layer
    imFG = ImageChops.multiply(im3,imBG)

    # enhance foreground image
    enhancerFG = ImageEnhance.Brightness(imFG)
    imFG = enhancerFG.enhance(fg_brightness)

    # compose the final image
    imFinal = Image.new('RGBA', imFG.size)
    imFinal.paste(imBG)

    if data_mask is not None:
        data_mask = data_mask + np.isnan(data)
        mask = make_mask_image(data_mask)

    else:
        data_mask = np.isnan(data)
        mask = make_mask_image(data_mask)

    imFinal.paste(imFG, mask=mask.resize(imFinal.size))
    im = ax.imshow(imFinal, extent=extent, origin=origin)

    try:
        return fig, ax, cb, im
    except NameError:
        try:
            return cb, im
        except NameError:
            im

def make_mask_image(mask):
    maskArr = np.array(Image.new('RGBA', (mask.shape[1],mask.shape[0])))
    maskArr[:,:,3] = ~mask*255
    return Image.fromarray(maskArr, 'RGBA')

