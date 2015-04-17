import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shutil import copyfile 

class sw4_image(object):
    
    """
    A class to read and hold sw4 image files, takes only a filename
    And optinally if the source is given in terms of velocity it is possible to set it to get correct units
    To get the best understanding of this class see the sw4 image file format in the user's guide
    """
    
    class patch_info(object):
        
        """
        A sub class of sw4_image, one instance is created per patch and holds it's data
        The instances are created automatically by read_patches_info(self) 
        The class takes an info list and holds it to the matching variables in each patch
        
        h - the patch cell size
        zmin -the smallest z in the patch (Note that z is increasing downwards)
        i,ni,j,nj - first and last values of the grid indices, ni and nj corspondes to the number of grid points in each direction, i,j normally equal 1
        """
        
        def __init__(self,info_list,plane):
            
            self.h = info_list[0]
            self.zmin = info_list[1]
            self.i = info_list[2]
            self.ni = info_list[3]
            self.j = info_list[4]
            self.nj = info_list[5]
            self.extent=[0,(self.ni-self.i)*self.h,-1*(self.nj-self.j)*self.h,0]
            if plane == 'z':self.extent=[0,(self.nj-self.j)*self.h,0,(self.ni-self.i)*self.h]
    
            
    def __init__(self,filename,velocity_source='no'):
        
        #set the file name
        self.filename=filename 
        self.velocity_source=velocity_source
        
        #read the header of the file
        (self.precision, self.n_of_patches, self.time, self.plane, self.location, self.mode, self.grid_info, self.creation_time, self.units) = self.readhdr()
        
        #read the patches info 
        (self.patches_info,position) = self.read_patches_info()
        
        #read the patches and remember position in the file
        (self.patches,position)=self.read_patches(61)
        
        #if grid data is stored read the grid data
        if self.grid_info == 1:
            self.grid_data=self.read_patches(position)[0]
            
        
        
    def readhdr(self):
        
        """
        A function to read thefile header
        The header is made of
        
        precision - int32
        number of patches - int32
        time - float64
        plane - int32
        location - float64
        mode - int32
        grid_info - int32
        creation_time - string of 24 bytes
        """
        
        #dicitoneris to translate header info to data
        precision_dict = {4:'float32',8:'float64'}
        
        plane_dict = {0:'x',1:'y',2:'z'}
        
        if self.velocity_source[0]=='n':

            mode_dict = {1:'X Displacment',2:'Y Displacment',3:'Z Displacment',4:'Density',5:'lambda',6:'mue',7:'P Waves Velocity',8:'S Waves Velocity',9:'uex(x)',10:'uex(y)',11:'uex(z)',
                        12:'divu',13:'curlu',14:'divu(t)',15:'curlu(t)',16:'lat',17:'lon',18:'topo',19:'x',20:'y',21:'z',
                        22:'uerror(x)',23:'uerror(y)',24:'uerror(z)',25:'Velocity Magnitude',26:'Horizontal Velocity',27:'Maximum Horizontal Velocity',
                        28:'Maximum Vertical Velocity',29:'Displacment Magnitude',30:'Horizontal Displacment Magnitude',31:'Maximum Horizontal Displacment',32:'Maximum Vertical Displacment'}

            units_dict = {1:'m',2:'m',3:'m',4:'gr/m^3',5:'lambda',6:'mue',7:'m/s',8:'m/s',9:'uex(x)',10:'uex(y)',11:'uex(z)',
                         12:'divu',13:'curlu',14:'divu(t)',15:'curlu(t)',16:'lat',17:'lon',18:'topo',19:'x',20:'y',21:'z',
                         22:'uerror(x)',23:'uerror(y)',24:'uerror(z)',25:'m/s',26:'m/s',27:'m/s',
                         28:'m/s',29:'m',30:'m',31:'m',32:'m'}
        else:
            mode_dict = {1:'X Velocity',2:'Y Velocity',3:'Z Velocity',4:'Density',5:'lambda',6:'mue',7:'P Waves Velocity',8:'S Waves Velocity',9:'uex(x)',10:'uex(y)',11:'uex(z)',
                        12:'divu',13:'curlu',14:'divu(t)',15:'curlu(t)',16:'lat',17:'lon',18:'topo',19:'x',20:'y',21:'z',
                        22:'uerror(x)',23:'uerror(y)',24:'uerror(z)',25:'Accleration Magnitude',26:'Horizontal Accleration',27:'Maximum Horizontal Accleration',
                        28:'Maximum Vertical Velocity',29:'Velocity Magnitude',30:'Horizontal Velocity Magnitude',31:'Maximum Horizontal Velocity',32:'Maximum Vertical Velocity'}

            units_dict = {1:'m/s',2:'m/s',3:'m/s',4:'gr/m^3',5:'lambda',6:'mue',7:'m/s',8:'m/s',9:'uex(x)',10:'uex(y)',11:'uex(z)',
                         12:'divu',13:'curlu',14:'divu(t)',15:'curlu(t)',16:'lat',17:'lon',18:'topo',19:'x',20:'y',21:'z',
                         22:'uerror(x)',23:'uerror(y)',24:'uerror(z)',25:'m/s^2',26:'m/s^2',27:'m/s^2',
                         28:'m/s^2',29:'m/s',30:'m/s',31:'m/s',32:'m/s'}

        
        
        f=open(self.filename,'rb')
        
        precision, n_of_patches = np.fromfile(f, dtype='int32', count=2)
        time = np.fromfile(f, dtype='float64', count=1)[0]
        plane = np.fromfile(f, dtype='int32', count=1)[0]
        location = np.fromfile(f, dtype='float64', count=1)[0]
        mode,grid_info = np.fromfile(f, dtype='int32', count=2)
        creation_time = str(f.read(24))
        
        if n_of_patches == 0: 
            print 'Error while reading file, 0 in number of patches'
            return None

        precision = precision_dict[precision]
        units= units_dict[mode]
        mode=mode_dict[mode]
        plane = plane_dict[plane]
        
        #if grid info is stored, the last patch is the grid info and should not be counted
        if grid_info == 1:
            n_of_patches-=1

        f.close()
        
        return precision, n_of_patches, time, plane, location, mode, grid_info, creation_time, units
        
    def read_patches_info(self):
        
        """
        A function to read the patches info
        Each patch has the foloowing data:
        h - float64
        zmin - float 64
        i,ni,j,nj - int32
        
        For more information about thier use see the patch_info class
        """
        
        f=open(self.filename,'rb')
        
        #61 is the length of the header, at this position the patches info start
        f.seek(61) 
        patches_info=[]
        
        #read the patches info, create an instance for each and store them in the list patches_info
        for i in range(self.n_of_patches):
            
            info_list = list(np.fromfile(f, dtype='float64', count=2))+list(np.fromfile(f, dtype='int32', count=4))
            patches_info.append(self.patch_info(info_list,self.plane))
        
        position=f.tell()
        f.close()
        return patches_info, position
        
        
    def read_patches(self,position):
        """
        A function to read the patches data
        Data is stored as a list of floats(each 32 or 64 bytes) in the length of ni*nj
        """
        f=open(self.filename,'rb')
        f.seek(position)
        patches = []
        
        #read the patches data, reshape each into a ni*nj array and store them in the list patches
        for i in range(self.n_of_patches):
            data = np.fromfile(f, dtype=self.precision, count=(self.patches_info[i].ni*self.patches_info[i].nj))
            patches.append(data.reshape(self.patches_info[i].nj,self.patches_info[i].ni))
        
        position=f.tell()
        return patches,position
    
    def plot(self,**imshow_kwargs):
            
        sw4_plot_image(self,**imshow_kwargs)
