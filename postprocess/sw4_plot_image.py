from seispy.postprocess.sw4_read_image import sw4_image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sw4_plot_image(img,figsize=(15,15),title='auto',xlabel='auto',ylabel='auto',colorbar_label='auto',
                   xmin=None,xmax=None,ymax=None,ymin=None,**imshow_kwargs):

    """
    The function Takes an sw4_image instance and plots it
    If not given the function will autumatically give size,title and labels
    The function supports all imshow kwargs

    The automatic titles assmumes the solution is displacment (source is Gaussian int etc.)
    If the solution is velocity titles wich says displacmnet are velocity and velocity are accleration
    """



    if not (isinstance(img , sw4_image)):
        print 'Cannot plot image, Input must be an sw4_image instance'
        return None

    if not('vmax' in imshow_kwargs): imshow_kwargs['vmax']=img.patches[0].std()
    if not('vmin' in imshow_kwargs): imshow_kwargs['vmin']=0
    if not('extent' in imshow_kwargs): imshow_kwargs['extent']=img.patches_info[0].extent


    if title == 'auto':
        title = img.mode


    if xlabel == 'auto':
        if img.plane == 'z': xlabel='Y distance [m]'
        elif img.plane == 'y': xlabel='X distance [m]'
        else: xlabel='Y distance [m]'

    if ylabel == 'auto':
        if img.plane == 'z': ylabel='X distance [m]'
        elif img.plane == 'y': ylabel='Z distance [m]'
        else: ylabel='Z distance [m]'


    if colorbar_label == 'auto':
        colorbar_label = '['+img.units+']'


    fig, ax = plt.subplots(figsize=figsize)

    data=img.patches[0]
    if img.plane=='z' :
        data=data.T
    f=ax.imshow(data,**imshow_kwargs)

    divider=make_axes_locatable(ax)
    cax=divider.append_axes("right",size="2%",pad=0.1)
    cbar = plt.colorbar(f,cax)
    cbar.ax.get_yaxis().labelpad = 20

    cbar.ax.set_ylabel(colorbar_label,rotation=270)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymax=1000)

    fig.show()
