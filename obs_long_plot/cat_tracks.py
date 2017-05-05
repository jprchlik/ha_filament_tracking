import create_plots as cp
#from descartes import PolygonPatch
from sunpy.sun import solar_semidiameter_angular_size
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from shapely.geometry import Polygon,MultiPoint
import astropy.units as u
from sunpy.physics import solar_rotation
from datetime import datetime,timedelta
from matplotlib.path import Path


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



rot_type = 'howard'

pickled = os.path.isfile('../init_data/FITracked_3yr.pic')
if pickled:#use pickle file if it already exits
    dat = pd.read_pickle('../init_data/FITracked_3yr.pic')
else: #create pickle file if doesnt exist
    infile = '../init_data/FITracked_3yr.txt'
    #dat = ascii.read('../init_data/FITracked_3yr.txt',delimiter='\t',guess=False)
    dat = pd.read_csv(infile,delimiter='\t')
    #add variables like datetime and average position to dat
    dat = ap.add_props(dat).dat
    dat.to_pickle('../init_data/FITracked_3yr.pic')


#testing purposes
#dat = dat[0:10]

#ha = cp.halpha_plot(dat,'dummy','dummy')

#get uniq track ids and convert to list so we can remove values as the are replaced
uniqlist = np.unique(dat['track_id'].values).tolist()

#cadence to look for max solar extent
dt = timedelta(minutes=20)

#for i in uniqlist:
ii = 0

#list of replaced tracks
replacelst = []
#loop over all uniq track names
for i in uniqlist:

    if i in replacelst: continue #skip track if already replaced

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
            curx, cury =  solar_rotation.rot_hpc(t1meanx*u.arcsec,t1meany*u.arcsec,t132b,maxt,rot_type=rot_type) #current position
            curr = np.sqrt(curx.value**2.+cury.value**2.)
    
            if maxt > t132b + timedelta(days=14): curr = 90000. #if the rotation goes for more than 14 days assume error and end search
            p += 1 
    
    
#do not let the distance in the mean y position change by more than square mag. distance of the shape
        dismag = np.sqrt((t1posx.max()-t1posx.min())**2+(t1posy.max()-t1posy.min())**2)
    #conditions to check for possible intersection
        c1t = dat['event_starttime_dt'].values > t1str #The time is greater than
        c2x = dat['meanx'] > t1meanx #the x position is greater than the x position of the first filament
        c3y = np.abs(dat['meany']-t1meany) < dismag #mean y value has not change more than the magniutde of the polygon
        c4t = dat['event_starttime_dt'].values < np.datetime64(maxt) #time not beyond when filament is on limb
        c5i = dat['track_id'].values != i #it is not already a part of the same track
     
    
        print '######################################################'
        print 'Track = {0:4d}'.format(i)
    #    print np.where(c1t)
    #    print np.where(c2x)
    #    print np.where(c3y)
    #    print np.where(c4t)
    #    print np.where(c5i)
        possm, = np.where((c1t) & (c2x) & (c3y) & (c4t) & (c5i)) 
    
        for j in possm:
             t232b = datetime.utcfromtimestamp(dat['event_starttime_dt'].values[j].tolist()/1e9) #covert to datetime object
             t12posx, t12posy =  solar_rotation.rot_hpc(t1posx*u.arcsec,t1posy*u.arcsec,t132b,t232b,rot_type=rot_type) #current position rotated to future time
    
             #remove units
             t12posx, t12posy = t12posx.value, t12posy.value 
     
             #convert 2 data points to array
             t2posx,t2posy = calc_poly_values(dat['hpc_bbox'].values[j])
           
             print 'Poss. Match with track = {0:4d}'.format(dat['track_id'].values[j])
    
    #convert coordinates to format for shapely readin
             c1fmt = [(t12posx[p],t12posy[p]) for p in range(len(t12posy))]
             c2fmt = [(t2posx[p],t2posy[p]) for p in range(len(t2posy))]
    
      
             poly1 = Polygon(c1fmt)
             poly2 = Polygon(c2fmt)
#get the shape of the intersections
             try:
                 poly3 = poly1.intersection(poly2)
                 #fractional area overlap
                 a1r = poly3.area/poly1.area 
                 a2r = poly3.area/poly2.area
                 print 'Area overlap with Prim = {0:4.3f}, with Sec = {1:4.3f}'.format(a1r,a2r)
             except ValueError:
                 print 'Could not calculate overlap'
                 continue
#dummy test plotting
#             fig, ax =plt.subplots()
#             ax.plot(t12posx,t12posy,color='blue')
#             ax.plot(t2posx,t2posy,color='red')
#             fig.savefig('test_track_{0:1d}_w_track_{1:1d}.png'.format(i,j))
            
 
#Switching to intersecting shape   
##             if poly1.intersects(poly2):
             alimit = .10 #must contain 10% of the area in intersection
             if ((a1r > alimit) | (a2r > alimit)):
                 #replace track number
                 print 'Match with track = {0:4d}'.format(dat['track_id'].values[j])
                 replace, = np.where(dat['track_id'].values == dat['track_id'].values[j])
                 dat['track_id'].values[replace] = i #replace matched track with previous id
                 if dat['track_id'].values[j] not in uniqlist: replacelst.append(dat['track_id'].values[j])
    

         


    ii += 1 #increment at end of loop

dat.to_pickle('concatentated_per_shape_3yr_file.pic')
