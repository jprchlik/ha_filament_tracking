import matplotlib
matplotlib.rc('font',size=24) #fix multiprocessing issue
matplotlib.use('agg')
import shapely 
import matplotlib.pyplot as plt
from shapely.wkt  import dumps, loads
from matplotlib.patches import *
from sunpy.physics import solar_rotation
import astropy.units as u
import pyfits
import datetime


class halpha_plot:

    def __init__(self,dat,ifile,tid,pdir):
        self.dat = dat
        self.ifile = ifile
        self.tid = tid
        self.pdir = pdir


  
    def  calc_poly_values(self,coor):
    #list of polygon coordinates
        corlist =  coor.replace('POLYGON((','').replace('))','').split(',')
    #create array and calculate mean x,y value for start
        corarra =  np.zeros((len(corlist),2))
        for j,k in enumerate(corlist):
            values = k.split(' ')
            corarra[j,0] = float(values[0])
            corarra[j,1] = float(values[1])

        return corarra[:,0],corarra[:,1]

    def plot_rotation(self,start,coor,dh=0,color='red',linestyle='-'):
    
        xs, ys = self.calc_poly_values(coor)
    #calculate the mean position
        stopx, stopy = solar_rotation.rot_hpc(xs*u.arcsec,ys*u.arcsec,start,self.stop)
    #get rid of units
        stopx, stopy = stopx.value, stopy.value
    
    
        self.ax.plot(stopx,stopy,linestyle=linestyle,color=color,zorder=500)
    #    ax.text(meanx,meany,'{0:6d}'.format(tid),color=color,fontsize=8)
    
    def plot_filament_track(self):

    
        dpi = 200
        good, = np.where(self.dat['track_id'].values == self.tid)
    
    #get start time of track for filename
#        ofname = '{0}_track{1:6d}'.format(dat['event_starttime'][good[0]],i).replace(' ','0').replace(':','_')
        self.ofile = self.ifile.split('/')[-1].replace('fits.fz','png')
        sun = pyfits.open(self.ifile)
#Solar Halpha data   
        sundat = sun[1].data
#get image extent (i.e. physical coordinates)
        x0 = sun[1].header['CRVAL1']-sun[1].header['CRPIX1']
        y0 = sun[1].header['CRVAL2']-sun[1].header['CRPIX2']

        dx = 1.#assumed approximate
        dy = 1.#assumed approximate
  
        sx, sy = np.shape(sundat)
    #add to stoptime (i.e. observed time) 
        self.stop = datetime.datetime.strptime(self.ofile[:-6],'%Y%m%d%H%M%S')
    
    #create figure and add sun
        self.fig, self.ax = plt.subplots(figsize=(7,7),dpi=dpi)

        self.ax.imshow(sundat,cmap=plt.cm.gray,extent=[x0,x0+dx*sx,y0+dy*sy,y0])
        self.ax.text(x0,y0,sun[1].header['DATE-OBS'],color='white',fontsize=18,fontweight='bold')
#        rs = plt.Circle((0.,0.),radius=1000.,color='gray',fill=False,linewidth=5,zorder=0)
#        ax.add_patch(rs)
    
    #plot track polygons for given id
        for j in good:
            inc = 'red'
            if self.dat['obs_observatory'].values[j] == 'HA2': inc='blue'
            poly = plt.Polygon(loads(self.dat['hpc_bbox'].values[j]).exterior,color=inc,linewidth=0.5,fill=None)
            self.ax.add_patch(poly)
    #over plot rotation track
        if good.size > 1:
     #array of time differences between obs and filament track
            td = np.abs(self.dat['event_starttime_dt'][good]-self.stop)
            nearest, = np.where(td == td.min())
            roplot = good[nearest][0] #nearest filament trace in time
        else: 
            roplot = good[0]
#plot rotation of nearest filament placement
        self.plot_rotation(self.dat['event_starttime_dt'][roplot],self.dat['hpc_bbox'].values[roplot],color='green')
    
    
    #Setup plots
        ticks = [-1000.,-500.,0.,500.,1000.]
        lim = [x0,x0+sx*dx]
        self.ax.set_xlim(lim)
        self.ax.set_ylim(lim)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.set_xlabel('Solar X [arcsec]')
        self.ax.set_ylabel('Solar Y [arcsec]')
    
    #save fig
        self.fig.savefig(self.pdir+self.ofile,bbox_pad=.1,bbox_inches='tight',dpi=dpi)
        self.fig.clear()
        plt.close()
    
