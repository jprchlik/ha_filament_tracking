import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
import shapely as sp
import shapely.geometry as geo
from shapely.wkt import dumps, loads
import sunpy.wcs
from astropy.io import fits as pyfits
from datetime import datetime, timedelta
import itertools
from scipy import stats
import statsmodels.api as sm

###############################################################################
################### Reading in initial filament stuff #########################
###############################################################################

### read in filament track file
#fil=pd.read_csv("concatentated_per_enlarged_shape_3yr_file.txt",delimiter="\t")
fila = pd.read_pickle("../obs_long_plot/concatentated_per_enlarged_shape_6yr_file_parellel.pic")
### relative base directory for pickle file
bdir = "../obs_long_plot/"
### pickle file name
#filn = "concatentated_per_enlarged_shape_3yr_file.pic"
#fil=pd.read_pickle(bdir+filn)
### converts string coordinates to shapely objects
fil['hpc_bbox'] = [loads(i) for i in fil['hpc_bbox'].values]


#put in lat and log coorindates
fil['meanx_hgs'] = 0
fil['meany_hgs'] = 0
#use centriod for meanx and meany values
for j,i in enumerate(fil['hpc_bbox']):
    fil['meanx'][j],fil['meany'][j] =i.centroid.x,i.centroid.y 
    fil['meanx_hgs'][j],fil['meany_hgs'][j] =sunpy.wcs.convert_hg_hpc(i.centroid.x,i.centroid.y)

###############################################################################
############## Creates plot of avg y value wrt # of tracks ####################
###############################################################################

u_tracks = np.unique(fil['track_id'])   # unique tracks, holds the number of the filament being tracked
n_tracks = np.zeros(u_tracks.size)      # number of instances of each track
y_tracks = np.zeros(u_tracks.size)      # average y value of each track

#average value of track throughout its time on disk
fil['track_y'] = 0.
#number of instances of filaments throughout its time on disk
fil['track_s'] = 0.

### find number of track instances, store in n_tracks
### find average y value of each track, store in y_tracks    
for i,j in enumerate(u_tracks):
    f_track, = np.where(fil['track_id'].values == j)    # get indices where track equals track id
    n_tracks[i] = f_track.size                          # store the number of track instances
    y_tracks[i] = np.mean(fil['meany'].values[f_track]) # calc approximate avg y value for each track
    fil['track_y'][f_track] = np.median(fil['meany'].values[f_track])
    fil['track_s'][f_track] = f_track.size
    

### create and save plot of avg y value as a fcn of number of tracks
### n_tracks vs. y_tracks
###############################################################################
####### Creates polygons for testing lat/long, establishes lat buffer #########
###############################################################################

buff = 100 # latitude buffer in arcsec
### converts 30 degrees HG to HPC
x,y = sunpy.wcs.convert_hg_hpc(0,30)
#print y
################################################################################# needs to be fixed
#x1,fil_mask = sunpy.wcs.convert_to_coord(0,60,HelioCentric,HelioProjective)
#print x1
#print fil_mask
fil_mask=780

sol_rad = 900 ### solar radius is approx 900 arcsec
### creates a north and south rectangle for +-40 degrees to the poles
n_rect = geo.box(-sol_rad, y, sol_rad, sol_rad, ccw=True)
s_rect = geo.box(-sol_rad, -y, sol_rad, -sol_rad, ccw=True)
test_rect = geo.box(-sol_rad, 0, sol_rad, buff, ccw=True)
### creates east/west limb boundaries  
### creating the 60 degree heliocentric latitude circle
east_circ_theta = np.linspace(0.5*np.pi,1.5*np.pi)
west_circ_theta = np.linspace(0.5*np.pi,-0.5*np.pi)
e_circ_x = fil_mask*np.cos(east_circ_theta)
e_circ_y = fil_mask*np.sin(east_circ_theta)
w_circ_x = fil_mask*np.cos(west_circ_theta)
w_circ_y = fil_mask*np.sin(west_circ_theta)
east_circ_pts = list()
west_circ_pts = list()
for angle in range(len(east_circ_theta)):
    east_circ_pts.append((e_circ_x[angle],e_circ_y[angle]))
east_circ_pts.append((0,-sol_rad))
east_circ_pts.append((-sol_rad,-sol_rad))
east_circ_pts.append((-sol_rad,sol_rad))
east_circ_pts.append((0,sol_rad))
for angle in range(len(west_circ_theta)):
    west_circ_pts.append((w_circ_x[angle],w_circ_y[angle]))
west_circ_pts.append((0,-sol_rad))
west_circ_pts.append((sol_rad,-sol_rad))
west_circ_pts.append((sol_rad,sol_rad))
west_circ_pts.append((0,sol_rad))
e_lat_circle = geo.Polygon(east_circ_pts)
w_lat_circle = geo.Polygon(west_circ_pts)

###############################################################################
############ Making an image overlay to verify placement of boxes #############
###############################################################################

ifile = "20120115140014Th.fits.fz"      # import sun image
sun = pyfits.open(ifile)
### solar H alpha data   
sundat = sun[1].data
### set up image properties
### get image radius in arcsec (i.e. physical coordinates)
x0 = sun[1].header['CRVAL1']-sun[1].header['CRPIX1']
y0 = sun[1].header['CRVAL2']-sun[1].header['CRPIX2']
    
### change per pixel in arcsec
dx = 1.
dy = 1.  
sf = 1. ### assume solar radius is 900. arcsec for now  
sx, sy = np.shape(sundat)

### create figure and add sun
fig, ax = plt.subplots()
### set sun to fill entire range
fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
### Turn off axis ticks
ax.set_axis_off()
ax.imshow(sundat,cmap=plt.cm.gray,extent=[sf*x0,sf*(x0+dx*sx),sf*y0,sf*(y0+dy*sy)],origin='lower',vmin=-900,vmax=5000)
north_box = plt.Polygon(n_rect.exterior,color="red",linewidth=1.0,fill=None)
ax.add_patch(north_box)
south_box = plt.Polygon(s_rect.exterior,color="red",linewidth=1.0,fill=None)
ax.add_patch(south_box)

e_circle = plt.Polygon(e_lat_circle.exterior,color="yellow",linewidth=1.0,fill=None)
ax.add_patch(e_circle)
w_circle = plt.Polygon(w_lat_circle.exterior,color="orange",linewidth=1.0,fill=None)
ax.add_patch(w_circle)

test_buff = plt.Polygon(test_rect.exterior,color="blue",linewidth=1.0,fill=None)
ax.add_patch(test_buff)
#fig.savefig("box_sun_overlay_test",bbox_pad=.1,bbox_inches='tight')
fig.savefig("current_visualization",bbox_pad=.1,bbox_inches='tight')

###############################################################################
############## Creates list of fil tracks that go limb to limb ################
###############################################################################

east_cut = []
west_cut = []

### 
for i,j in enumerate(fil['hpc_bbox']):
    ### if the filament intersects either the north or south rectangle, continue
    if fil['hpc_bbox'][i].intersects(n_rect) or fil['hpc_bbox'][i].intersects(s_rect):
        ### if the filament intersects the east or west limb, record the track id in a new list
        if fil['hpc_bbox'][i].intersects(e_lat_circle):
            east_cut.append(fil['track_id'][i])
        elif fil['hpc_bbox'][i].intersects(w_lat_circle):
            west_cut.append(fil['track_id'][i])

### test to see the track ids that occur in both east_cut and west_cut
### those that do are poleward of +- 30 degrees and go limb to limb
good_ids = set(east_cut).intersection(west_cut)
good_ids = list(good_ids)
good_ids.sort()
#print good_ids
#print len(good_ids)

###############################################################################
######## Read in and organize/create cavity lat/long data in dataframe ########
###############################################################################

### read in cavity data for eastern limb cavities
cav=pd.read_csv("cavitylist_date_eastonly.txt",delim_whitespace=True)

### converts coords from HG to HPC
cav_start_lat_hpc = []
cav_end_lat_hpc = []
buff_start_lat_hpc = []
buff_end_lat_hpc = []
cav_start_long_hpc = []
cav_end_long_hpc = []

for i,j in enumerate(cav['start_lat']):
    x1,y1 = sunpy.wcs.convert_hg_hpc(cav['start_long'][i],j)
    x2,y2 = sunpy.wcs.convert_hg_hpc(cav['end_long'][i],cav['end_lat'][i])
    cav_start_lat_hpc.append(y1)
    cav_end_lat_hpc.append(y2)
    if cav_start_lat_hpc[i] > cav_end_lat_hpc[i]:
        buff_start_lat_hpc.append(cav_start_lat_hpc[i]+buff) ### creates a buffered latitude (50 arcsec)
        buff_end_lat_hpc.append(cav_end_lat_hpc[i]-buff)   ### for when matching fil to cav by latitude
    else:
        buff_start_lat_hpc.append(cav_start_lat_hpc[i]-buff) ### creates a buffered latitude (50 arcsec)
        buff_end_lat_hpc.append(cav_end_lat_hpc[i]+buff)
    cav_start_long_hpc.append(x1)
    cav_end_long_hpc.append(x2)
    
cav['start_long_hpc'] = cav_start_long_hpc
cav['end_long_hpc'] = cav_end_long_hpc

### make latitudes negative if in the southern hemisphere
for i,j in enumerate(cav['cavity_id']):
    if 'S' in j:
        cav['start_lat'][i] = -1*cav['start_lat'][i]
        cav['end_lat'][i] = -1*cav['end_lat'][i]
        cav_start_lat_hpc[i] = -1*cav_start_lat_hpc[i]
        cav_end_lat_hpc[i] = -1*cav_end_lat_hpc[i]
        buff_start_lat_hpc[i] = -1*buff_start_lat_hpc[i]
        buff_end_lat_hpc[i] = -1*buff_end_lat_hpc[i]
cav['start_lat_hpc'] = cav_start_lat_hpc
cav['end_lat_hpc'] = cav_end_lat_hpc
cav['buff_start_lat_hpc'] = buff_start_lat_hpc
cav['buff_end_lat_hpc'] = buff_end_lat_hpc

###############################################################################
################ Create datetime objects in cavity dataframe ##################
###############################################################################

### making cav and fil start/end times datetime objects
### when matching filaments to cavities, allow a 12 hour buffer
#NOT NEEDED BECAUSE EVENT_STARTTIME_DT AND EVENT_ENDTIME_DT are already properly formatted in pickle file 2019/04/16 J. Prchlik
###fil_start_ymd = []
###for i,j in enumerate(fil['event_starttime_dt']):
###    #By using pickle file event_starttime_dt is a timestamp not txt 
###    #fil_start_ymd.append(datetime.strptime(fil['event_starttime_dt'][i], '%Y-%m-%d %H:%M:%S'))
###    fil_start_ymd.append(fil['event_starttime_dt'][i]))
###    #fil_start_ymd[i] = fil_start_ymd[i].replace(hour=0, minute=0, second=0)
###fil['event_starttime_dt'] = fil_start_ymd
###fil_end_ymd = []
###for i,j in enumerate(fil['event_endtime_dt']):
###    #By using pickle file event_endtime_dt is a timestamp not txt 
###    fil_end_ymd.append(fil['event_endtime_dt'][i])
###    #fil_end_ymd[i] = fil_end_ymd[i].replace(hour=0, minute=0, second=0)
###fil['event_endtime_dt'] = fil_end_ymd

cav_start_ymd = []
for i,j in enumerate(cav['start_date']):
    cav_start_ymd.append(datetime.strptime(cav['start_date'][i], '%Y/%m/%d_%H:%M'))
cav['start_date_dt'] = cav_start_ymd
cav_end_ymd = []
for i,j in enumerate(cav['end_date']):
    cav_end_ymd.append(datetime.strptime(cav['end_date'][i], '%Y/%m/%d_%H:%M'))
cav['end_date_dt'] = cav_end_ymd

### creates new columns in fil for total track start/end times, and number of instances of each fil
track_start = [0]*len(fil['event_starttime_dt'])
track_end = [0]*len(fil['event_starttime_dt'])
num_inst = [0]*len(fil['event_starttime_dt'])

for i,j in enumerate(fil['track_id']):
    if track_start[i] == 0:
        id_indices, = np.where(fil['track_id'] == j)
        for x,y in enumerate(id_indices):
            track_start[y] = min(fil['event_starttime_dt'][id_indices])
            track_end[y] = max(fil['event_endtime_dt'][id_indices])
            num_inst[y] = len(id_indices)
        
fil['track_start'] = track_start
fil['track_end'] = track_end
fil['num_inst'] = num_inst

###############################################################################
### Creates dictionaries and lists for matching fil to cav in each category ###
###############################################################################

### dictionaries and lists to define categories
fil['cat_id'] = 0
cat1 = {}
cat2 = {}
cat3 = {}
cat4 = []
cat5 = []
### length of time buffer in hours
hours_buff = 24

### cat1: needs fil from good_ids, and cavities that last 14 days or more
### loop through cavities
for x,y in enumerate(cav['start_date_dt']):
    ### clear fil_list before each cavity
    fil_list = []
    ### creates box for cavity that extends across the sun
    cav_box = geo.box(-sol_rad, cav['buff_end_lat_hpc'][x], sol_rad, cav['buff_start_lat_hpc'][x], ccw=True)
    ### makes sure cavity goes limb to limb
    if (cav_end_ymd[x] - y).days >= 14:
        ### loop through filaments
        for i,j in enumerate(fil['track_start']):
            #Passing logic for catigory id 
            #track goes limb to limb
            goodtrack = fil['track_id'][i] in good_ids
            #filament track is on the west limb the same time as a cavity
            ### if the filament start time is between the cavity start time + buffer
            ### and the cavity end time + buffer, check the latitude.
            cav_tmat = j >= (y - timedelta(hours=hours_buff)) and j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff))
            #filament is in same location as the cavity
            ### make sure the filament occurs at the same latitude as the cavity
            cav_pmat = fil['hpc_bbox'][i].intersects(cav_box)
            if ((goodtrack) & (cav_tmat) & (cav_pmat)):
                fil_list.append(fil['track_id'][i])     
                fil['cat_id'][i] = 1
        ### if there are any filaments in the list, attach them to the corresponding cavity
        if len(fil_list) > 0:
            cat1[cav['cavity_id'][x]] = np.unique(fil_list)

### cat2: needs fil from good_ids, cavities that last less than 14 days
### loop through cavities
for x,y in enumerate(cav['start_date_dt']):
    ### ensure cavities were not already placed in cat 1 to avoid duplicates (should be redundant)
    if cav['cavity_id'][x] not in cat1:
        ### clear fil_list before each cavity
        fil_list = []
        ### creates box for cavity that extends across the sun
        cav_box = geo.box(-sol_rad, cav['buff_end_lat_hpc'][x], sol_rad, cav['buff_start_lat_hpc'][x], ccw=True)
        ### makes sure that cavity does not go limb to limb
        if (cav_end_ymd[x] - y).days < 14:
            ### loop through filaments
            for i,j in enumerate(fil['event_starttime_dt']):
                #Passing logic for catigory id 
                #track goes limb to limb
                goodtrack = fil['track_id'][i] in good_ids
                #filament track is on the west limb the same time as a cavity
                ### if the filament start time is between the cavity start time + buffer
                ### and the cavity end time + buffer, check the latitude.
                cav_tmat = j >= (y - timedelta(hours=hours_buff)) and j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff))
                #filament is in same location as the cavity
                ### make sure the filament occurs at the same latitude as the cavity
                cav_pmat = fil['hpc_bbox'][i].intersects(cav_box)
                if ((goodtrack) & (cav_tmat) & (cav_pmat)):
                    fil_list.append(fil['track_id'][i])     
                    fil['cat_id'][i] = 2
            ### if there are any filaments in the list, attach them to the corresponding cavity
            if len(fil_list) > 0:
                cat2[cav['cavity_id'][x]] = np.unique(fil_list)

### cat3: needs fil to not be from good_ids, cavities that last less than 14 days
for x,y in enumerate(cav['start_date_dt']):
    ### ensure cavities were not already placed in cat 1/2 to avoid duplicates (should be redundant)
    if cav['cavity_id'][x] not in cat1 and cav['cavity_id'][x] not in cat2:
        ### clear fil_list before each cavity
        fil_list = []
        ### creates box for cavity that extends across the sun
        cav_box = geo.box(-sol_rad, cav['buff_end_lat_hpc'][x], sol_rad, cav['buff_start_lat_hpc'][x], ccw=True)
        ### makes sure that cavity does not go limb to limb
        if (cav_end_ymd[x] - y).days < 14:
            for i,j in enumerate(fil['event_starttime_dt']):
                #Passing logic for catigory id 
                #track goes limb to limb
                goodtrack = fil['track_id'][i] not in good_ids
                #filament track is on the west limb the same time as a cavity
                ### if the filament start time is between the cavity start time + buffer
                ### and the cavity end time + buffer, check the latitude.
                cav_tmat = ((j >= (y - timedelta(hours=hours_buff))) and (j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff))))
                #filament is in same location as the cavity
                ### make sure the filament occurs at the same latitude as the cavity
                cav_pmat = fil['hpc_bbox'][i].intersects(cav_box)
                #only include somewhat stable tracks in category 3
                cav_inst = fil['num_inst'][i] > 3
                if ((goodtrack) & (cav_tmat) & (cav_pmat) & (cav_inst)):
                    fil_list.append(fil['track_id'][i])     
                    fil['cat_id'][i] = 3
            ### if there are any filaments in the list, attach them to the corresponding cavity
        if len(fil_list) > 0:
            cat3[cav['cavity_id'][x]] = np.unique(fil_list)

### cat4: needs fil with no matching cavs
for i,j in enumerate(fil['track_id']):
    ### if the filament intersects either the north or south rectangle, continue
    if fil['hpc_bbox'][i].intersects(n_rect) or fil['hpc_bbox'][i].intersects(s_rect):
        ### only include filaments that have more than 5 instances in its track
        if fil['num_inst'][i] > 5:
            ### search dictionaries for filaments that do not correspond to a cavity
            if j not in [x1 for v1 in cat1.values() for x1 in v1] and j not in [x2 for v2 in cat2.values() for x2 in v2] and j not in [x3 for v3 in cat3.values() for x3 in v3]:
                cat4.append(j)
                fil['cat_id'][i] = 4
### list of unique fil that have more than 3 instances and no matching cav
cat4 = np.unique(cat4)

### cat5: needs cav with no matching fil
for x,y in enumerate(cav['cavity_id']):
    if y not in cat1 and y not in cat2 and y not in cat3:
        cat5.append(y) 
 

print(len(cat1))
print(len(cat2))
print(len(cat3))
print(len(cat4))
print(len(cat5))

#%%
###############################################################################
############### Columns for median length, tilt, and latitude #################
###############################################################################

### create a column in fil for the median length of each fil track
med_length = [0]*len(fil['event_starttime_dt'])
med_y = [0]*len(fil['event_starttime_dt'])
med_y_hpc = [0]*len(fil['event_starttime_dt'])
med_x_hpc = [0]*len(fil['event_starttime_dt'])
med_tilt = [0]*len(fil['event_starttime_dt'])
for i,j in enumerate(fil['track_id']):
    if med_length[i] == 0:
        id_indices, = np.where(fil['track_id'] == j)
        for x,y in enumerate(id_indices):
            med_length[y] = np.median(fil['fi_length'][id_indices])
            med_y[y] = np.median(fil['meany'][id_indices])
            med_y_hpc[y] = np.median(fil['meany_hgs'][id_indices])
            med_x_hpc[y] = np.median(fil['meanx_hgs'][id_indices])
            med_tilt[y] = np.median(fil['fi_tilt'][id_indices])

fil['med_length'] = med_length

### create a column in fil for the median tilt of each fil track
fil['med_tilt'] = med_tilt

### create a column in fil for the median latitude of each fil track
fil['med_y'] = med_y
fil['med_y_hpc'] = med_y_hpc
fil['med_x_hpc'] = med_x_hpc



#Get a counts for all categories
fil.sort(['track_id','cat_id'],inplace=True)


fil.to_pickle('filament_catagories_7yr.pic')


#get counts of array
c_df = fil[~fil.index.duplicated(keep='last')]
c_df['c_tracks'] = 1
print(c_df.groupby('cat_id')['c_tracks'].sum().reset_index())

#get all filament ids in category to check with what is store in array
in1 = []
for i in cat1.keys(): 
    for p in cat1[i].tolist(): in1.append(p)



in2 = []
for i in cat2.keys(): 
    for p in cat2[i].tolist(): in2.append(p)


in3 = []
for i in cat3.keys(): 
    for p in cat3[i].tolist(): in3.append(p)


#check category matching
c_df.loc[in1,['track_id','cat_id']]
c_df.loc[in2,['track_id','cat_id']]
c_df.loc[in3,['track_id','cat_id']]
c_df.loc[cat4,['track_id','cat_id']]
