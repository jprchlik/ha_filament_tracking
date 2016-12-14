import matplotlib
matplotlib.rc('font',size=24) #fix multiprocessing issue
matplotlib.use('agg')
import shapely 
import matplotlib.pyplot as plt
from shapely.wkt  import dumps, loads
from matplotlib.patches import *
from sunpy.physics import solar_rotation
from sunpy.sun import solar_semidiameter_angular_size
import astropy.units as u
import pyfits
import datetime


class halpha_plot:
    """ A class which creates an image from an input file.
        Currently used to create H alpha GONG images."""

    def __init__(self,dat,ifile,pdir):
        """Initializes variables to be used by the halpha_plot class.

           This function just creates the object to be class later.

           Parameters
           ----------
           dat  :  pandas object
              The pandas object is the read in track data file FITtracked_3yr.txt
           ifile: string
              ifile is the input GONG H alpha file
           pdir : string
              pdir is a string which points to the directory for plotting
         
           Returns
           -------
           obj
              Returns initialized halpha_plot class
        """
        self.dat = dat
        self.ifile = ifile
        self.pdir = pdir


  
    def calc_poly_values(self,coor):
        """calc_poly_values returns an array of x,y values from Polygon string in pandas object
         
           calc_poly_values parses a POLYGON string from the track pandas object by splitting on commas 
           and assuming coordiantes are in the order x1,y1,x2,y2,ect

           Parameters
           ----------
           coor : string
               POLYGON string from FITracked_3yr.txt
           
           Returns
           -------
           x : numpy array
               x values of Polygon coordinates
           y : numpy array
               y values of Polygon coordinates
        """

    #list of polygon coordinates
        corlist =  coor.replace('POLYGON((','').replace('))','').split(',')
    #create array and calculate mean x,y value for start
        corarra =  np.zeros((len(corlist),2))
        for j,k in enumerate(corlist):
            values = k.split(' ')
            corarra[j,0] = float(values[0])
            corarra[j,1] = float(values[1])

        return corarra[:,0],corarra[:,1]

    def plot_rotation(self,start,coor,dh=0,color='red',linestyle='-',alpha=0.5):
        """
        plot_rotation overplots a h alpha filament track accounting for solar roation

        The function plot_rotation uses the current image time to correct the track observation time for solar rotation.
        To correct for rotation the function uses the solar_rotation module from sunpy

        Parameters
        ----------
        start: datetime object
            start is the observed time of the track.
        coor : str
            coor is the polygon string for the track's shape.
        dh   : datetime object
            dh is deprecated, thus no longer used by plot_rotation (default = 0)
        color: str
            color is the numpy string color to used for outlining the track (default = 'red').
        linestyle: str
            linstyle is the matplotlib string to use for the line (default = '-')
        alpha: float
            alpha is the opacity of the line to plot (default = 0.5, range = [0.0,1.0])

        Returns
        -------
        None
       
        """
    
        xs, ys = self.calc_poly_values(coor)
    #calculate the mean position
        stopx, stopy = solar_rotation.rot_hpc(xs*u.arcsec,ys*u.arcsec,start,self.stop)
    #get rid of units
        stopx, stopy = stopx.value, stopy.value
    
    
        self.ax.plot(stopx,stopy,linestyle=linestyle,color=color,zorder=500,alpha=alpha)
    #    ax.text(meanx,meany,'{0:6d}'.format(tid),color=color,fontsize=8)
    
    def plot_filament_track(self):

        """
        Function to create H alpha GONG plots with tracks overplotted

        plot_filament_tracks creates png files of GONG halpha data with all observed H alpha filament tracks visible on the sun at that time.
        The function can only be called after initialization of the halpha_plot object. 


        Parameters
        ----------
        self

        Returns
        -------
        H alpha with with tracks overplotted

        """ 

    
    #get start time of track for filename
#        ofname = '{0}_track{1:6d}'.format(dat['event_starttime'][good[0]],i).replace(' ','0').replace(':','_')
        self.ofile = self.ifile.split('/')[-1].replace('fits.fz','png')
        try:
            sun = pyfits.open(self.ifile)
        # observed time of GONG halpha image
            self.stop = datetime.datetime.strptime(self.ofile[:-6],'%Y%m%d%H%M%S')
    
    
    #Solar Halpha data   
            sundat = sun[1].data
    #Set up image properties
            sc  = 1 
            dpi = 100*sc
    #get image extent (i.e. physical coordinates)
            x0 = sun[1].header['CRVAL1']-sun[1].header['CRPIX1']
            y0 = sun[1].header['CRVAL2']-sun[1].header['CRPIX2']
    
            dx = 1.#assumed approximate
            dy = 1.#assumed approximate
    
    #Correct for Halpha images automatic scaling of solar radius to 900``
    #        try:
    #            asr = solar_semidiameter_angular_size(self.dat['event_starttime'].values[good[0]])
    #        except:
    #            asr = solar_semidiameter_angular_size(self.dat['event_starttime'].values[good])
            asr = solar_semidiameter_angular_size(self.stop.strftime('%Y/%m/%d %H:%M:%S'))
    
    #store the scale factor
            sf = asr.value/900.
      
            sx, sy = np.shape(sundat)
        #get tracks visible on the surface using a given time frame
            good, = np.where((self.dat['total_event_start'] <= self.stop) & (self.dat['total_event_end'] >= self.stop))
        
        #create figure and add sun
            self.fig, self.ax = plt.subplots(figsize=(sc*float(sx)/dpi,sc*float(sy)/dpi),dpi=dpi)
        #set sun to fill entire range
            self.fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    #Turn off axis
            self.ax.set_axis_off()
    
            self.ax.imshow(sundat,cmap=plt.cm.gray,extent=[sf*x0,sf*(x0+dx*sx),sf*y0,sf*(y0+dy*sy)],origin='lower')
    #text offset
            poff = 0.01
            self.ax.text(sf*x0+poff*sf*(dx*sx-x0),sf*y0+poff*sf*(dy*sy-y0),sun[1].header['DATE-OBS'],color='white',fontsize=38,fontweight='bold')
    #        rs = plt.Circle((0.,0.),radius=1000.,color='gray',fill=False,linewidth=5,zorder=0)
    #        ax.add_patch(rs)
    
        
        #plot track polygons for given id
            for j in good:
                inc = 'red'
                if self.dat['obs_observatory'].values[j] == 'HA2': inc='blue'
                poly = plt.Polygon(loads(self.dat['hpc_bbox'].values[j]).exterior,color=inc,linewidth=0.5,fill=None)
                self.ax.add_patch(poly)
                xx, yy = self.calc_poly_values(self.dat['hpc_bbox'].values[j])
                self.ax.text(np.mean(xx),np.max(yy),self.dat['track_id'].values[j],alpha=.5,color=inc)
    
    #list of track ids
            ltid = np.unique(self.dat['track_id'].values[good])
    
    #loop over track ids to find the closest rotational track
            for tid in ltid:
            #over plot rotation track
                idmatch, = np.where(self.dat['track_id'].values == tid)
                if idmatch.size > 1:
             #array of time differences between obs and filament track
                    td = np.abs(self.dat['event_starttime_dt'][idmatch]-self.stop)
                    nearest, = np.where(td == td.min())
                    roplot = idmatch[nearest][0] #nearest filament trace in time
                else: 
                    roplot = idmatch[0]
        #plot rotation of nearest filament placement
                self.plot_rotation(self.dat['event_starttime_dt'][roplot],self.dat['hpc_bbox'].values[roplot],color='green')
        
        
        #Setup plots
    #        ticks = [-1000.,-500.,0.,500.,1000.]
            lim = [sf*x0,sf*(x0+sx*dx)]
            self.ax.set_xlim(lim)
            self.ax.set_ylim(lim)
           
    #remove axis labels just make an image
    #        self.ax.set_xticks(ticks)
    #        self.ax.set_yticks(ticks)
    #        self.ax.set_xlabel('Solar X [arcsec]')
    #        self.ax.set_ylabel('Solar Y [arcsec]')
        
        #save fig
            self.fig.savefig(self.pdir+self.ofile,bbox_inches=0,dpi=dpi)
            self.fig.clear()
            plt.close()
    
        except:
            print 'Unable to create image'
