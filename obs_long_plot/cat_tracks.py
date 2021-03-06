import create_plots as cp
import make_gong_cat
#from descartes import PolygonPatch
from sunpy.sun import solar_semidiameter_angular_size
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from shapely.geometry import Polygon,MultiPoint,LineString
import astropy.units as u
#function deprecated
#from sunpy.physics import solar_rotation

#use new sunpy rotation routine
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from astropy.coordinates import SkyCoord
#get frame for coordiantes
from sunpy.coordinates import frames
import astropy.units as u




from datetime import datetime,timedelta
from matplotlib.path import Path


#mimic program rot_hpc with new routines
def rot_hpc(xs,ys,start,end,rot_type='meaningless'):
    #xs, ys = calc_poly_values(coor)
    #update deprecated function J. Prchlik 2017/11/03
    c = SkyCoord(xs,ys,obstime=start,frame=frames.Helioprojective)                    
    #rotate start points to end time
    nc = solar_rotate_coordinate(c,end)

    #split into x and y
    rotx, roty = nc.Tx, nc.Ty

    #return rotated coordinates
    return rotx,roty

def calc_poly_values(coor):
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


file_name = '../init_data/FITracked.txt'
pick_name = file_name.replace('.txt','.pic')

rot_type = 'howard'

pickled = os.path.isfile(pick_name)
if pickled:#use pickle file if it already exits
    dat = pd.read_pickle(pick_name)
else: #create pickle file if doesnt exist
    #dat = ascii.read('../init_data/FITracked_3yr.txt',delimiter='\t',guess=False)
    dat = pd.read_csv(file_name,delimiter='\t')
    #add variables like datetime and average position to dat
    dat = ap.add_props(dat).dat
    dat.to_pickle(pick_name)


#testing purposes
#dat = dat[0:10]

#ha = cp.halpha_plot(dat,'dummy','dummy')

#get uniq track ids and convert to list so we can remove values as the are replaced
uniqlist = np.unique(dat['track_id'].values).tolist()

#cadence to look for max solar extent
dt = timedelta(minutes=120)

#for i in uniqlist:
ii = 0

#list of replaced tracks
replacelst = []
#loop over all uniq track names
for i in uniqlist:

    if i in replacelst:
        continue #skip track if already replaced

#get where pandas dataframe equals track id
    track, = np.where(dat['track_id'].values == i)

#which element to grab from track
    for k in track:

        t1str = dat['event_starttime_dt'].values[k]
    
    #grab data from first element in track
        t132b = datetime.utcfromtimestamp(t1str.tolist()/1e9)
        t1posx, t1posy = calc_poly_values(dat['hpc_bbox'].values[k])
        t1meanx = dat['meanx'].values[k]
        t1meany = dat['meany'].values[k]
    
        maxr = solar_semidiameter_angular_size(t132b).value-10. #10 arcsec of the limb
        
        curr = np.sqrt(t1meanx**2.+t1meany**2.)
        p = 1
    
    #find when filament goes over the limb
        while curr < maxr:
            maxt = t132b+timedelta(minutes=20*p)  
            #remove deprecated fucntion J. Prchlik 2017/11/03
            #curx, cury =  solar_rotation.rot_hpc(t1meanx*u.arcsec,t1meany*u.arcsec,t132b,maxt,rot_type=rot_type) #current position
            curx, cury =  rot_hpc(t1meanx*u.arcsec,t1meany*u.arcsec,t132b,maxt,rot_type=rot_type) #current position
            
            curr = np.sqrt(curx.value**2.+cury.value**2.)
    
            if maxt > t132b + timedelta(days=14):
                curr = 90000. #if the rotation goes for more than 14 days assume error and end search

            p += 1 
    
    
#do not let the distance in the mean y position change by more than square mag. distance of the shape
        dismag = np.sqrt((t1posx.max()-t1posx.min())**2+(t1posy.max()-t1posy.min())**2)
    #conditions to check for possible intersection
        c1t = dat['event_starttime_dt'].values > t1str #The time is greater than
        c2x = dat['meanx'] > t1meanx-10. #the x position is greater than the x position of the first filament with a 10arcsec pad (approximately an hour)
        c3y = np.abs(dat['meany']-t1meany) < dismag #mean y value has not change more than the magniutde of the polygon
        c4t = dat['event_starttime_dt'].values < np.datetime64(maxt) #time not beyond when filament is on limb
        c5i = dat['track_id'].values != i #it is not already a part of the same track
     
    
        print('######################################################')
        print('Track = {0:4d}'.format(i))
        #cut track to try and match
        possm, = np.where((c1t) & (c2x) & (c3y) & (c4t) & (c5i)) 

    
        for j in possm:
             t232b = datetime.utcfromtimestamp(dat['event_starttime_dt'].values[j].tolist()/1e9) #covert to datetime object
             #removing depricated function J. Prchlik (2017/11/03)
             #t12posx, t12posy =  solar_rotation.rot_hpc(t1posx*u.arcsec,t1posy*u.arcsec,t132b,t232b,rot_type=rot_type) #current position rotated to future time
             t12posx, t12posy =  rot_hpc(t1posx*u.arcsec,t1posy*u.arcsec,t132b,t232b,rot_type=rot_type) #current position rotated to future time

    
             #remove units
             t12posx, t12posy = t12posx.value, t12posy.value 
     
             #convert 2 data points to array
             t2posx,t2posy = calc_poly_values(dat['hpc_bbox'].values[j])
           
             print('Poss. Match with track = {0:4d}'.format(dat['track_id'].values[j]))
    
             #convert coordinates to format for shapely readin
             c1fmt = [(t12posx[p],t12posy[p]) for p in range(len(t12posy))]
             c2fmt = [(t2posx[p],t2posy[p]) for p in range(len(t2posy))]

    
      

             #increase the "real" filament size by 20% to find near neighbors 
             scale = 3.0 
            
             #get the shape of the intersections
             try:
                 poly1 = Polygon(c1fmt)
                 polyl = LineString(c1fmt)
                 poly2 = Polygon(c2fmt)
                 #use the large shape for area intersection
                 polyb = Polygon(polyl.parallel_offset(scale,'right',join_style=1)) # draw a 20% larger round polygon
                 poly3 = polyb.intersection(poly2)
                 #fractional area overlap
                 a1r = poly3.area/poly1.area 
                 a2r = poly3.area/poly2.area
                 print('Area enlargement = {0:5.4f}'.format(polyb.area/poly1.area))
                 print('Area overlap with Prim = {0:4.3f}, with Sec = {1:4.3f}'.format(a1r,a2r))
             except:
                 print('Could not calculate overlap')
                 continue
            
 
             #Switching to intersecting shape   
             alimit = 0.05 #must contain 5% of the area in intersection
             if ((a1r > alimit) | (a2r > alimit)):
                 #replace track number
                 print('Match with track = {0:4d}'.format(dat['track_id'].values[j]))
                 replace, = np.where(dat['track_id'].values == dat['track_id'].values[j])
                 dat['track_id'].values[replace] = i #replace matched track with previous id
                 if dat['track_id'].values[j] not in uniqlist:
                     replacelst.append(dat['track_id'].values[j])
    

         


    ii += 1 #increment at end of loop

outf = 'concatentated_per_enlarged_shape_6yr_file_parellel.pic'
dat.to_pickle(outf)

#make_gong_cat.main(outf,'filament_tracking_enlarged_5_per.mp4')
