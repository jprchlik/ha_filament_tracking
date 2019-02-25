import matplotlib
import sys
matplotlib.rc('font',size=24) #fix multiprocessing issue
matplotlib.use('agg')
import shapely 
import numpy as np
import matplotlib.pyplot as plt
from shapely.wkt  import dumps, loads
from matplotlib.patches import *
#function deprecated 
#from sunpy.physics import solar_rotation

#use new sunpy rotation routine
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from astropy.coordinates import SkyCoord
#get frame for coordiantes
from sunpy.coordinates import frames
import astropy.units as u
import os
import sunpy
from sunpy.cm import cm

#get a median filter for wavelet transformation
from matplotlib.colors import Normalize
from scipy.signal import medfilt

#J. Prchlik 2018/05/02
import pywt
#Interested in the Stationary Wavelet Transformation or a Trous or starlet (swt2)

from glob import glob
import get_synoptic_files as gsf

from sunpy.sun import solar_semidiameter_angular_size
import astropy.units as u
from astropy.io import fits as  pyfits
import datetime


#J. Prchlik 2018/05/02
#Added to give physical coordinates
def img_extent(img):
    """
    Give physical cooridnates for image mapping to physical coordinates
    """
# get the image coordinates in pixels
    px0 = img.meta['crpix1']
    py0 = img.meta['crpix2']
# get the image coordinates in arcsec 
    ax0 = img.meta['crval1']
    ay0 = img.meta['crval2']
# get the image scale in arcsec 
    axd = img.meta['cdelt1']
    ayd = img.meta['cdelt2']
#get the number of pixels
    tx,ty = img.data.shape
#get the max and min x and y values
    minx,maxx = px0-tx,tx-px0
    miny,maxy = py0-ty,ty-py0
#convert to arcsec
    maxx,minx = maxx*axd,minx*axd
    maxy,miny = maxy*ayd,miny*ayd

    return maxx,minx,maxy,miny


class halpha_plot:
    """ A class which creates an image from an input file.
        Currently used to create H alpha GONG images."""

    def __init__(self,dat,ifile,pdir,lref=False,dpi=100,ext='png',add_aia=False,aia_wav=[193]):
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
           lref: Boolean, optional
              reference line for good tracks to match with filament (30 degrees, Default = False)
           dpi:  int, optional
              Dots Per Inch of output image (Default = 100)
           ext: string. optional
              File extension of output image (Default = 'png')
           add_aia: Boolean, optional
              Add AIA image to outer edge (Default = False)
           aia_wav: list
              List of wavelengths to download (Default = [193])
         
           Returns
           -------
           obj
              Returns initialized halpha_plot class
        """
        self.dat = dat
        self.ifile = ifile
        self.pdir = pdir
        self.lref = lref
        self.dpi  = dpi
        self.ext  = ext
        self.add_aia= add_aia
        self.aia_wav = aia_wav

    def draw_lines(self):
        """
           Draw a line at 40 degrees latitude 
           Parameters
           ----------
           self

           Returns
           -------
           None
        """
        r = self.asr.value
        y = 475.
        x = (r**2-y**2)**(.5)
        xs = np.linspace(-x,x,10)
        yt = np.zeros(xs.size)+y
        yb = np.zeros(xs.size)-y
        self.ax.plot(xs,yt,'-.',color='red',alpha=1.,linewidth=5,zorder=5000)
        self.ax.plot(xs,yb,'-.',color='red',alpha=1.,linewidth=5,zorder=5000)
        

    def add_aia_image(self):

        file_fmt = '{0:%Y/%m/%d/H%H00/AIA%Y%m%d_%H%M_}'
        tries = 4
        wave = self.aia_wav
         
        gsf.download(self.stop-datetime.timedelta(minutes=tries),self.stop+datetime.timedelta(minutes=tries),
                     datetime.timedelta(minutes=1),'',nproc=1,
                     syn_arch='http://jsoc.stanford.edu/data/aia/synoptic/',
                     f_dir=file_fmt,d_wav=wave)

        #Decide which AIA file to overplot
        nofile = True
        run = 0
        while nofile:
            testfile = file_fmt.format(self.stop+datetime.timedelta(minutes=run))+'{0:04d}.fits'.format(wave[0])
            #exit once you find the file
            if os.path.isfile(testfile):
                filep = testfile
                nofile = False
            run += 1
            #exit after tries
            if run == tries+1:
                nofile = False
                return

        img = sunpy.map.Map(filep)
        #Block add J. Prchlik (2016/10/06) to give physical coordinate values 
        #return extent of image
        maxx,minx,maxy,miny = img_extent(img)

        #Add wavelet transformation J. Prchlik 2018/01/17
        wav_img = img.data
        d_size = 15
        #get median filter
        n_back = medfilt(wav_img,kernel_size=d_size)
        #subtract median filter
        img_sub = wav_img-n_back

        #Use Biorthogonal Wavelet
        wavelet = 'bior2.6'

        #use 6 levels
        n_lev = 6
        o_wav = pywt.swt2(img_sub, wavelet, level=n_lev )
        #only use the first 4
        f_img = pywt.iswt2(o_wav[0:4],wavelet)
        #Add  wavelet back into image
        f_img = f_img+wav_img

        #remove zero values
        f_img[f_img < 0.] = 0.


        #set alpha values only works with png file 2018/05/02 J. Prchlik
        #alphas = np.ones(f_img.shape)
        #alphas[:,:] = np.linspace(1,0,f_img.shape[0])
        colors = Normalize((15.)**0.25,(3500.)**0.25,clip=True)
        #img_193 = cmap((f_img)**0.25) 
        img_193 = cm.sdoaia193(colors((f_img)**0.25))


        #get radius values and convert to arcsec
        mesh_x2, mesh_y2 = np.meshgrid(np.arange(img.data.shape[0]),np.arange(img.data.shape[1]))
        mesh_x2 = mesh_x2.T*(maxx-minx)/f_img.shape[0]+minx
        mesh_y2 = mesh_y2.T*(maxy-miny)/f_img.shape[1]+miny

        #mask out less than a solar radius
        rsun = sunpy.sun.solar_semidiameter_angular_size(t=img.meta['date-obs'][:-1].replace('T',' ')).value
        r2 = np.sqrt(mesh_x2**2+mesh_y2**2)/rsun

        rmin = .98
        rfad = 1.02
        rep2 = (r2 < rmin) 
        rep3 = ((r2 > rmin) & (r2 < rfad))
        img_193[...,3][rep2] = 0
        img_193[...,3][rep3] = (r2[rep3]-rmin)/(rfad-rmin)

        #plot the image in matplotlib
        self.ax.imshow(img_193,interpolation='none',cmap=cm.sdoaia193,origin='lower',vmin=(15.)**0.25,vmax=(3500.)**0.25,extent=[minx,maxx,miny,maxy])



  
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

    def plot_rotation(self,start,coor,dh=0,color='teal',linestyle='-',alpha=1.0,ids=None):
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
        self
       
        """
    
        xs, ys = self.calc_poly_values(coor)
        #calculate the mean position
        #stopx, stopy = solar_rotation.rot_hpc(xs*u.arcsec,ys*u.arcsec,start,self.stop)
        #update deprecated function J. Prchlik 2017/11/03
        c = SkyCoord(xs*u.arcsec,ys*u.arcsec,obstime=start,frame=frames.Helioprojective)                    
        #rotate start points to end time
        nc = solar_rotate_coordinate(c,self.stop)

        #get rid of units
        stopx, stopy = nc.Tx.value, nc.Ty.value
    
    
        self.ax.plot(stopx,stopy,linestyle=linestyle,color=color,zorder=500,alpha=alpha,linewidth=3)
        self.ax.text(np.mean(stopx),np.max(stopy),str(ids),alpha=1.,color=color,fontweight='bold')
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
        #replaced with named extension 2018/04/02 J. Prchlik
        self.ofile = self.ifile.split('/')[-1].replace('fits.fz',self.ext)
        try:
            sun = pyfits.open(self.ifile)
        # observed time of GONG halpha image
            self.stop = datetime.datetime.strptime(self.ofile[:-6],'%Y%m%d%H%M%S')
    
    
    #Solar Halpha data   
            sundat = sun[1].data
    #Set up image properties
            sc  = self.dpi/100.
            dpi = self.dpi
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
            self.asr = solar_semidiameter_angular_size(self.stop.strftime('%Y/%m/%d %H:%M:%S'))
    
    #store the scale factor
            sf = self.asr.value/900.
      
            sx, sy = np.shape(sundat)
        #get tracks visible on the surface using a given time frame
            good, = np.where((self.dat['total_event_start'] <= self.stop) & (self.dat['total_event_end'] >= self.stop))
        
        #create figure and add sun
            self.fig, self.ax = plt.subplots(figsize=(sc*float(sx)/dpi,sc*float(sy)/dpi),dpi=dpi)
        #set sun to fill entire range
            self.fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    #Turn off axis
            self.ax.set_axis_off()
    
            stat_max = 4300.
            stat_min = 75.
            self.ax.imshow(sundat,cmap=plt.cm.gray,extent=[sf*x0,sf*(x0+dx*sx),sf*y0,sf*(y0+dy*sy)],origin='lower',vmin=stat_min,vmax=stat_max)
    #text offset
            poff = 0.01
            self.ax.text(sf*x0+poff*sf*(dx*sx-x0),sf*y0+poff*sf*(dy*sy-y0),sun[1].header['DATE-OBS'],color='white',fontsize=38,fontweight='bold')
    #        rs = plt.Circle((0.,0.),radius=1000.,color='gray',fill=False,linewidth=5,zorder=0)
    #        ax.add_patch(rs)
            if self.lref: self.draw_lines()
    
        
#remove identifying plot
########        #plot track polygons for given id
########            for j in good:
########                inc = 'red'
########                if self.dat['obs_observatory'].values[j] == 'HA2': inc='blue'
########                poly = plt.Polygon(loads(self.dat['hpc_bbox'].values[j]).exterior,color=inc,linewidth=0.5,fill=None)
########                self.ax.add_patch(poly)
########                xx, yy = self.calc_poly_values(self.dat['hpc_bbox'].values[j])
########                self.ax.text(np.mean(xx),np.max(yy),self.dat['track_id'].values[j],alpha=.5,color=inc)
########    
    #list of track ids
            ltid = np.unique(self.dat['track_id'].values[good])
    
    #loop over track ids to find the closest rotational track
            for tid in ltid:
            #over plot rotation track
                idmatch, = np.where(self.dat['track_id'].values == tid)
                td = np.abs(self.dat['event_starttime_dt'][idmatch]-self.stop)
                nearest,= np.where(td == td.min())
                roplot = idmatch[nearest]
              ####  if idmatch.size < -100:
             #####array of time differences between obs and filament track
              ####      td = np.abs(self.dat['event_starttime_dt'][idmatch]-self.stop)
              ####      nearest, = np.where(td == td.min())
              ####      roplot = idmatch[nearest]#[0] #nearest filament trace in time
              ####  else: 
              ####      roplot = idmatch[0]
            
                roplot= roplot.tolist()
        #plot rotation of nearest filament placement
                for k in roplot: self.plot_rotation(self.dat['event_starttime_dt'][k],self.dat['hpc_bbox'].values[k],color='teal',ids=self.dat['track_id'].values[k])
        
        
        #Setup plots
    #        ticks = [-1000.,-500.,0.,500.,1000.]
            lim = [sf*x0,sf*(x0+sx*dx)]
            self.ax.set_xlim(lim)
            self.ax.set_ylim(lim)



            #Add AIA plot 2018/05/02 J. Prchlik
            if self.add_aia: self.add_aia_image()
           
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
            print sys.exc_info()
