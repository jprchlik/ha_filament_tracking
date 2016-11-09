import matplotlib
matplotlib.rc('font',size=24)
#fix multiprocessing issue
matplotlib.use('agg')
import sunpy
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import shapely
from shapely.wkt import dumps, loads
import os
import datetime
from fancy_plot import fancy_plot
import pandas as pd
from multiprocessing import Pool
from matplotlib.patches import *
from sunpy.physics import solar_rotation
import astropy.units as u


def calc_mean_pos(coor):
#list of polygon coordinates
    corlist =  coor.replace('POLYGON((','').replace('))','').split(',')
#create array and calculate mean x,y value for start
    corarra =  np.zeros((len(corlist),2))
    for j,k in enumerate(corlist):
        values = k.split(' ')
        corarra[j,0] = float(values[0])
        corarra[j,1] = float(values[1])

#find mean x and y values
    meanx, meany = np.mean(corarra[:,0]),np.mean(corarra[:,1])

    return meanx, meany

def plot_rotation(coor,start,stop,tid,ax,linestyle='-',dh=0,color='black'):

#add to stoptime 
    addstop = datetime.timedelta(hours=dh)

#calculate the mean position
    meanx, meany = calc_mean_pos(coor)

    stopx, stopy = solar_rotation.rot_hpc(meanx*u.arcsec,meany*u.arcsec,start,stop+addstop)
#get rid of units
    stopx, stopy = stopx.value, stopy.value
    

    ax.plot([meanx,stopx],[meany,stopy],linestyle=linestyle,color=color)
    ax.text(meanx,meany,'{0:6d}'.format(tid),color=color,fontsize=8)


    return ax
    

def plot_filament_track(i):
    global dat
   
    dpi = 200
    good, = np.where(dat['track_id'].values == i)

#get start time of track for filename
    ofname = '{0}_track{1:6d}'.format(dat['event_starttime'][good[0]],i).replace(' ','0').replace(':','_')

#create figure and add sun
    fig, ax = plt.subplots(figsize=(7,7),dpi=dpi)
    rs = plt.Circle((0.,0.),radius=1000.,color='gray',fill=False,linewidth=5,zorder=0)
    ax.add_patch(rs)

#plot track polygons for given id
    for j in good:
        inc = 'red'
        if dat['obs_observatory'].values[j] == 'HA2': inc='blue'
        poly = plt.Polygon(loads(dat['hpc_bbox'].values[j]).exterior,color=inc,linewidth=0.5,fill=None)
        ax.add_patch(poly)
#over plot rotation track
        #ax = plot_rotation(dat['hpc_coord'].values[j],dat['event_starttime_dt'][j],dat['event_endtime_dt'][j],dat['track_id'].values[j],ax,linestyle='--',dh=12,color='red')

#find other nearby events in time
    nearbyh = 12. # hours
    nearbya = 10. # arcsec
    nearstartt = np.abs(dat['event_starttime_dt'][good[0]]-dat['event_endtime_dt'])  <= datetime.timedelta(hours=nearbyh)
    nearendt   = np.abs(dat['event_endtime_dt'][good[-1]]-dat['event_starttime_dt']) <= datetime.timedelta(hours=nearbyh)

    nearstartp = np.sqrt((dat['meanx'].values[good[0]] -dat['meanx'].values)**2+(dat['meany'].values[good[0]] -dat['meany'].values)**2) <= nearbya
    nearendp   = np.sqrt((dat['meanx'].values[good[-1]]-dat['meanx'].values)**2+(dat['meany'].values[good[-1]]-dat['meany'].values)**2) <= nearbya

#:   nearby, = np.where((nearstartt & nearstartp) | (nearendt & nearendp))
    nearby, = np.where((nearendp & nearendt))
    print dat['meanx'].values[good[-1]]
    print dat['meanx'][nearby]
  

#plot within time tracks
    for j in nearby:
        poly = plt.Polygon(loads(dat['hpc_bbox'].values[j]).exterior,color='black',linewidth=0.5,fill=None)
        ax.add_patch(poly)
        ax = plot_rotation(dat['hpc_coord'].values[j],dat['event_starttime_dt'][j],dat['event_endtime_dt'][j],dat['track_id'].values[j],ax,dh=48)
        

#Setup plots
    fancy_plot(ax)
    ticks = [-1000.,-500.,0.,500.,1000.]
    lim = [-1200.,1200.]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('Solar X [arcsec]')
    ax.set_ylabel('Solar Y [arcsec]')

#save fig
    fig.savefig('track_plots/'+ofname,bbox_pad=.1,bbox_inches='tight',dpi=dpi)
    fig.clear()
    plt.close()

    return

infile = '../init_data/FITracked_3yr.txt'
#dat = ascii.read('../init_data/FITracked_3yr.txt',delimiter='\t',guess=False)
dat = pd.read_csv(infile,delimiter='\t')



#date format
dfmt = '%Y-%m-%dT%H:%M:%S'

#create a date time array
dtstr = ['start','end']
for j in dtstr:
    dat['event_{0}time_dt'.format(j)] = [ datetime.datetime.strptime(i,dfmt) for i in dat['event_{0}time'.format(j)]]

#dat['meanx'],dat['meany'] =  [ calc_mean_pos(coor) for coor in dat['hpc_bbox']]
vals =  [ calc_mean_pos(coor) for coor in dat['hpc_bbox']]
vals = np.array(vals)

dat['meanx'] =vals[:,0]
dat['meany'] =vals[:,1]

#sort by date
dat = dat.sort(['event_starttime_dt','event_endtime_dt'],ascending=[1,1])



#array containing track IDs
rows = np.unique(dat['track_id'].values)



#rows = rows[:2]
#for z in rows:
#    print z
#
#    plot_filament_track(z)

nproc = 10

pool = Pool(processes=nproc)

out = pool.map(plot_filament_track,rows)

pool.close()






