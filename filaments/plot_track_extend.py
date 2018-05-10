import numpy as np
import os
import sunpy
import sunpy.map
import get_synoptic_files as gsf
import pandas as pd
import matplotlib.pyplot as plt
from sunpy import sun
from shapely.geometry import Polygon

from matplotlib.colors import Normalize

from sunpy.cm import cm

from sunpy.sun import solar_semidiameter_angular_size
import astropy.units as u
from astropy.io import fits as  pyfits
import datetime

from scipy.signal import medfilt
#J. Prchlik 2018/05/02
import pywt
#Interested in the Stationary Wavelet Transformation or a Trous or starlet (swt2)

def draw_lines(asr,ax): 
    """
       Draw a line at 40 degrees latitude 
       Parameters
       ----------
       self

       Returns
       -------
       None
    """
    r = asr.value
    y = 475.
    x = (r**2-y**2)**(.5)
    xs = np.linspace(-x,x,10)
    yt = np.zeros(xs.size)+y
    yb = np.zeros(xs.size)-y
    ax.plot(xs,yt,'-.',color='red',alpha=1.,linewidth=2,zorder=5000)
    ax.plot(xs,yb,'-.',color='red',alpha=1.,linewidth=2,zorder=5000)
    return ax


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



def add_aia_image(stime,ax,tries=4):

    file_fmt = '{0:%Y/%m/%d/H%H00/AIA%Y%m%d_%H%M_}'
    tries = 4
    wave = [193]

    gsf.download(stime,stime+datetime.timedelta(minutes=tries),
                 datetime.timedelta(minutes=1),'',nproc=1,
                 syn_arch='http://jsoc.stanford.edu/data/aia/synoptic/',
                 f_dir=file_fmt,d_wav=wave)

    #Decide which AIA file to overplot
    nofile = True
    run = 0
    while nofile:
        testfile = file_fmt.format(stime+datetime.timedelta(minutes=run))+'{0:04d}.fits'.format(wave[0])
        #exit once you find the file
        if os.path.isfile(testfile):
            filep = testfile
            nofile = False
        run += 1
        #exit after tries
        if run == tries+1:
            nofile = False

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
    ax.imshow(img_193,interpolation='none',cmap=cm.sdoaia193,origin='lower',vmin=(15.)**0.25,vmax=(3500.)**0.25,extent=[minx,maxx,miny,maxy])
    return ax




#track ID to plot
track_id = 12963
track_id = 12397
track_id = 14122
track_sp = 6840

fil = pd.read_pickle('filament_categories_hgs_mean_l.pic')

pfil = fil.loc[track_id,:]

#another track for the south pol
sfil = fil.loc[track_sp,:]

sr = sun.solar_semidiameter_angular_size()

r1 = plt.Circle((0,0),radius=sr.value,color='lightgray',fill=True,linewidth=5,zorder=0)




fig, ax =plt.subplots(figsize=(10,10))

#draw reference lines
draw_lines(sr,ax)

#plot all polygons 
for j,i in enumerate(pfil['hpc_bbox'].values):
     x,y = i.exterior.xy
     ax.plot(x,y,color='black',linewidth=2)
     if j == 0: 
         ax.text(np.mean(x),max(y)+100.,'Track ID = {0:1d}'.format(track_id),fontsize=20)
#plot all polygons  for sp track
for j,i in enumerate(sfil['hpc_bbox'].values):
     x,y = i.exterior.xy
     ax.plot(x,y,color='black',linewidth=2)
     if j == 5: 
         ax.text(-600,-650.,'Track ID = {0:1d}'.format(track_sp),fontsize=20)

add_aia_image(pd.to_datetime(pfil.event_starttime_dt.values[0]),ax)

ax.text(-1000,-1000,pfil.event_starttime.values[5],color='white',fontsize=24,fontweight='bold')
ax.add_patch(r1)
ax.set_axis_off()
ax.set_xlim([-1.1*sr.value,1.1*sr.value])
ax.set_ylim([-1.1*sr.value,1.1*sr.value])
fig.savefig('plots/track_evolution_{0:06d}.png'.format(track_id),bbox_pad=.1,bbox_inches='tight')
fig.savefig('plots/track_evolution_{0:06d}.eps'.format(track_id),bbox_pad=.1,bbox_inches='tight')
plt.show()