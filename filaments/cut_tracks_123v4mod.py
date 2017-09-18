import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
import shapely as sp
import shapely.geometry as geo
from shapely.wkt import dumps, loads
import sunpy.wcs
import pyfits
from datetime import datetime, timedelta
import itertools
from scipy import stats
import statsmodels.api as sm

###############################################################################
################### Reading in initial filament stuff #########################
###############################################################################

### read in filament track file
fil=pd.read_csv("concatentated_per_enlarged_shape_3yr_file.txt",delimiter="\t")
### relative base directory for pickle file
bdir = "../obs_long_plot/"
### pickle file name
filn = "concatentated_per_enlarged_shape_3yr_file.pic"
#fil2=pd.read_pickle(bdir+filn)
### converts string coordinates to shapely objects
fil['hpc_bbox_p'] = [loads(i) for i in fil['hpc_bbox'].values]

###############################################################################
############## Creates plot of avg y value wrt # of tracks ####################
###############################################################################

u_tracks = np.unique(fil['track_id'])   # unique tracks, holds the number of the filament being tracked
n_tracks = np.zeros(u_tracks.size)      # number of instances of each track
y_tracks = np.zeros(u_tracks.size)      # average y value of each track

### find number of track instances, store in n_tracks
### find average y value of each track, store in y_tracks    
for i,j in enumerate(u_tracks):
    f_track, = np.where(fil['track_id'].values == j)    # get indices where track equals track id
    n_tracks[i] = f_track.size                          # store the number of track instances
    y_tracks[i] = np.mean(fil['meany'].values[f_track]) # calc approximate avg y value for each track

"""
### create and save plot of avg y value as a fcn of number of tracks
### n_tracks vs. y_tracks
graph1, ax1 = plt.subplots()
ax1.plot(n_tracks, y_tracks,'bo')
ax1.set_xlabel('Number of Tracks')
ax1.set_ylabel('Average y-Value (arcseconds)')
ax1.set_title('Before 40 Degree Cuts')
graph1.savefig("n_tracks_vs_avg_y.png",bbox_pad=.1,bbox_inches='tight')
"""
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
for i,j in enumerate(fil['hpc_bbox_p']):
    ### if the filament intersects either the north or south rectangle, continue
    if fil['hpc_bbox_p'][i].intersects(n_rect) or fil['hpc_bbox_p'][i].intersects(s_rect):
        ### if the filament intersects the east or west limb, record the track id in a new list
        if fil['hpc_bbox_p'][i].intersects(e_lat_circle):
            east_cut.append(fil['track_id'][i])
        elif fil['hpc_bbox_p'][i].intersects(w_lat_circle):
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
fil_start_ymd = []
for i,j in enumerate(fil['event_starttime_dt']):
    fil_start_ymd.append(datetime.strptime(fil['event_starttime_dt'][i], '%Y-%m-%d %H:%M:%S'))
    #fil_start_ymd[i] = fil_start_ymd[i].replace(hour=0, minute=0, second=0)
fil['event_starttime_dt'] = fil_start_ymd
fil_end_ymd = []
for i,j in enumerate(fil['event_endtime_dt']):
    fil_end_ymd.append(datetime.strptime(fil['event_endtime_dt'][i], '%Y-%m-%d %H:%M:%S'))
    #fil_end_ymd[i] = fil_end_ymd[i].replace(hour=0, minute=0, second=0)
fil['event_endtime_dt'] = fil_end_ymd

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
            ### only search the tracks that go limb to limb
            if fil['track_id'][i] in good_ids:
                ### if the filament start time is between the cavity start time + buffer
                ### and the cavity end time + buffer, check the latitude.
                if j >= (y - timedelta(hours=hours_buff)) and j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff)):
                    ### make sure the filament occurs at the same latitude as the cavity
                    if fil['hpc_bbox_p'][i].intersects(cav_box):
                        fil_list.append(fil['track_id'][i])     
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
                ### only search tracks that go limb to limb
                if fil['track_id'][i] in good_ids:
                    ### if the filament start time is between the cavity start time + buffer
                    ### and the cavity end time + buffer, check the latitude.
                    if j >= (y - timedelta(hours=hours_buff)) and j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff)):
                        ### make sure the filament occurs at the same latitude as the cavity
                        if fil['hpc_bbox_p'][i].intersects(cav_box):
                            fil_list.append(fil['track_id'][i])     
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
            ### loop through filaments
            for i,j in enumerate(fil['event_starttime_dt']):
                ### only search tracks that do not go limb to limb
                if fil['track_id'][i] not in good_ids:
                    ### if the filament start time is between the cavity start time + buffer
                    ### and the cavity end time + buffer, check the latitude.
                    if j >= (y - timedelta(hours=hours_buff)) and j <= (cav['end_date_dt'][x] + timedelta(hours=hours_buff)):
                        ### make sure the filament occurs at the same latitude as the cavity
                        if fil['hpc_bbox_p'][i].intersects(cav_box):
                            fil_list.append(fil['track_id'][i])        
            ### if there are any filaments in the list, attach them to the corresponding cavity
            if len(fil_list) > 0:
                cat3[cav['cavity_id'][x]] = np.unique(fil_list)

### cat4: needs fil with no matching cavs
for i,j in enumerate(fil['track_id']):
    ### if the filament intersects either the north or south rectangle, continue
    if fil['hpc_bbox_p'][i].intersects(n_rect) or fil['hpc_bbox_p'][i].intersects(s_rect):
        ### only include filaments that have more than 3 instances in its track
        if fil['num_inst'][i] > 5:
            ### search dictionaries for filaments that do not correspond to a cavity
            if j not in [x1 for v1 in cat1.values() for x1 in v1] and j not in [x2 for v2 in cat2.values() for x2 in v2] and j not in [x3 for v3 in cat3.values() for x3 in v3]:
                cat4.append(j)
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
for i,j in enumerate(fil['track_id']):
    if med_length[i] == 0:
        id_indices, = np.where(fil['track_id'] == j)
        for x,y in enumerate(id_indices):
            med_length[y] = np.median(fil['fi_length'][id_indices])
fil['med_length'] = med_length

### create a column in fil for the median tilt of each fil track
med_tilt = [0]*len(fil['event_starttime_dt'])
for i,j in enumerate(fil['track_id']):
    if med_tilt[i] == 0:
        id_indices, = np.where(fil['track_id'] == j)
        for x,y in enumerate(id_indices):
            med_tilt[y] = np.median(fil['fi_tilt'][id_indices])
fil['med_tilt'] = med_tilt

### create a column in fil for the median latitude of each fil track
med_y = [0]*len(fil['event_starttime_dt'])
for i,j in enumerate(fil['track_id']):
    if med_y[i] == 0:
        id_indices, = np.where(fil['track_id'] == j)
        for x,y in enumerate(id_indices):
            med_y[y] = np.median(fil['meany'][id_indices])
fil['med_y'] = med_y

###############################################################################
############# Length, tilt, and latitude lists for each category ##############
###############################################################################

###############################################################################
################################# Category 1 ##################################
###############################################################################
### will hold the lengths and tilts for cat1
lengths1 = []
tilts1 = []
abs_tilts1 = []
lats1 = []

### create a list of filaments from the cat1 dict (id_nums)
id_nums = []
for track in cat1:
    id_nums.append(cat1[track])
id_nums = list(itertools.chain(*id_nums))

### if the filament is from the cat1 dict (id_nums), find its indices and save one of them in tot_indices
tot_indices = []
for i,j in enumerate(fil['track_id']):
    if j in id_nums:
        id_indices, = np.where(fil['track_id'] == j)
        tot_indices.append(min(id_indices))
### tot_indices contains indices, one per filament track in cat 1; create lists of median lengths/tilts
for i,j in enumerate(fil['track_id']):
    if i in tot_indices:
        lengths1.append(fil['med_length'][i])
        tilts1.append(fil['med_tilt'][i])
        abs_tilts1.append(abs(fil['med_tilt'][i]))
        lats1.append(fil['med_y'][i])
        
### altering lats1 due to nans
lats1_alt = []
tilts1_alt = []
lengths1_alt = []
for i,j in enumerate(lats1):
    if str(j) != 'nan':
        lats1_alt.append(j)
        tilts1_alt.append(tilts1[i])
        lengths1_alt.append(lengths1[i])

lats1_deg = []
for i,j in enumerate(lats1_alt):
    x,y = sunpy.wcs.convert_hpc_hg(0,j)
    lats1_deg.append(y)

### creates north and south tilt lists
tilts1_n = []
tilts1_s = []
lengths1_n = []
lengths1_s = []

abs_lats1 = []
for i,j in enumerate(lats1_deg):
    abs_lats1.append(abs(j))
    if j>0: ### northern hemisphere
        tilts1_n.append(tilts1[i])
        lengths1_n.append(lengths1[i])
    else:
        tilts1_s.append(tilts1[i])
        lengths1_s.append(lengths1[i])

abs_tilts1_n = []
abs_tilts1_s = []

for i,j in enumerate(tilts1_n):
    abs_tilts1_n.append(abs(j))
for i,j in enumerate(tilts1_s):
    abs_tilts1_s.append(abs(j))
    
###############################################################################
################################# Category 2 ##################################
###############################################################################        
### will hold the lengths and tilts for cat2
lengths2 = []
tilts2 = []
abs_tilts2 = []
lats2 = []

### create a list of filaments from the cat2 dict (id_nums)
id_nums = []
for track in cat2:
    id_nums.append(cat2[track])
id_nums = list(itertools.chain(*id_nums))

### if the filament is from the cat2 dict (id_nums), find its indices and save one of them in tot_indices
tot_indices = []
for i,j in enumerate(fil['track_id']):
    if j in id_nums:
        id_indices, = np.where(fil['track_id'] == j)
        tot_indices.append(min(id_indices))
### tot_indices contains indices, one per filament track in cat 1; create lists of median lengths/tilts
for i,j in enumerate(fil['track_id']):
    if i in tot_indices:
        lengths2.append(fil['med_length'][i])
        tilts2.append(fil['med_tilt'][i])
        abs_tilts2.append(abs(fil['med_tilt'][i]))
        lats2.append(fil['med_y'][i])
        
### altering lats2 due to nans
lats2_alt = []
tilts2_alt = []
lengths2_alt = []
for i,j in enumerate(lats2):
    if str(j) != 'nan':
        lats2_alt.append(j)
        tilts2_alt.append(tilts2[i])
        lengths2_alt.append(lengths2[i])

lats2_deg = []
for i,j in enumerate(lats2_alt):
    x,y = sunpy.wcs.convert_hpc_hg(0,j)
    lats2_deg.append(y)

### creates north and south tilt lists
tilts2_n = []
tilts2_s = []
lengths2_n = []
lengths2_s = []

abs_lats2 = []
for i,j in enumerate(lats2_deg):
    abs_lats2.append(abs(j))
    if j>0: ### northern hemisphere
        tilts2_n.append(tilts2[i])
        lengths2_n.append(lengths2[i])
    else:
        tilts2_s.append(tilts2[i])
        lengths2_s.append(lengths2[i])
        
abs_tilts2_n = []
abs_tilts2_s = []

for i,j in enumerate(tilts2_n):
    abs_tilts2_n.append(abs(j))
for i,j in enumerate(tilts2_s):
    abs_tilts2_s.append(abs(j))
    
###############################################################################
################################# Category 3 ##################################
###############################################################################         
### will hold the lengths and tilts for cat3
lengths3 = []
tilts3 = []
abs_tilts3 = []
lats3 = []

### create a list of filaments from the cat3 dict (id_nums)
id_nums = []
for track in cat3:
    id_nums.append(cat3[track])
id_nums = list(itertools.chain(*id_nums))

### if the filament is from the cat3 dict (id_nums), find its indices and save one of them in tot_indices
tot_indices = []
for i,j in enumerate(fil['track_id']):
    if j in id_nums:
        id_indices, = np.where(fil['track_id'] == j)
        tot_indices.append(min(id_indices))
### tot_indices contains indices, one per filament track in cat 3; create lists of median lengths/tilts
for i,j in enumerate(fil['track_id']):
    if i in tot_indices:
        lengths3.append(fil['med_length'][i])
        tilts3.append(fil['med_tilt'][i])
        abs_tilts3.append(abs(fil['med_tilt'][i]))
        lats3.append(fil['med_y'][i])
        
### altering lats1 due to nans
lats3_alt = []
tilts3_alt = []
lengths3_alt = []
for i,j in enumerate(lats3):
    if str(j) != 'nan':
        lats3_alt.append(j)
        tilts3_alt.append(tilts3[i])
        lengths3_alt.append(lengths3[i])

lats3_deg = []
for i,j in enumerate(lats3_alt):
    x,y = sunpy.wcs.convert_hpc_hg(0,j)
    lats3_deg.append(y)

### creates north and south tilt lists
tilts3_n = []
tilts3_s = []
lengths3_n = []
lengths3_s = []

abs_lats3 = []
for i,j in enumerate(lats3_deg):
    abs_lats3.append(abs(j))
    if j>0: ### northern hemisphere
        tilts3_n.append(tilts3[i])
        lengths3_n.append(lengths3[i])
    else:
        tilts3_s.append(tilts3[i])
        lengths3_s.append(lengths3[i])
    
###############################################################################
############################### Categories 123 ################################
###############################################################################         
### create a list of cat1, 2, and 3 lengths together
lengths123 = lengths1 + lengths2 + lengths3
tilts123 = tilts1 + tilts2 + tilts3
abs_tilts123 = abs_tilts1 + abs_tilts2 + abs_tilts3
lats123_nan = lats1 + lats2 + lats3

tilts123_n = tilts1_n + tilts2_n + tilts3_n
tilts123_s = tilts1_s + tilts2_s + tilts3_s

lengths123_n = lengths1_n + lengths2_n + lengths3_n
lengths123_s = lengths1_s + lengths2_s + lengths3_s

abs_tilts123_n = abs_tilts1_n + abs_tilts2_n + abs_tilts3_n
abs_tilts123_s = abs_tilts1_s + abs_tilts2_s + abs_tilts3_s

### some of the mean y data was just 'nan'
### remove the nans and create altered tilt and length lists so I can make scatter plots
#lats123 = [x for x in lats123_nan if str(x) != 'nan'] 
lats123 = []
tilts123_alt = []
lengths123_alt = []
for i,j in enumerate(lats123_nan):
    if str(j) != 'nan':
        lats123.append(j)
        tilts123_alt.append(tilts123[i])
        lengths123_alt.append(lengths123[i])
    
lats123_deg = []
for i,j in enumerate(lats123):
    x,y = sunpy.wcs.convert_hpc_hg(0,j)
    lats123_deg.append(y)

abs_lats123_deg = []
for i,j in enumerate(lats123_deg):
    abs_lats123_deg.append(abs(j))

###############################################################################
################################# Category 4 ##################################
###############################################################################
### create lists of fil lengths/tilts from cat4
lengths4 = []
tilts4 = []
abs_tilts4 = []
lats4_nan = []
for i,j in enumerate(cat4):
    id_indices, = np.where(fil['track_id'] == j)
    lengths4.append(fil['med_length'][min(id_indices)])
    tilts4.append(fil['med_tilt'][min(id_indices)])
    abs_tilts4.append(abs(fil['med_tilt'][min(id_indices)])) 
    lats4_nan.append(fil['med_y'][min(id_indices)])
    
### some of the mean y data was just 'nan'
### remove the nans and create altered tilt and length lists so I can make scatter plots
#lats4 = [x for x in lats4a if str(x) != 'nan']
lats4 = []
tilts4_alt = []
lengths4_alt = []
for i,j in enumerate(lats4_nan):
    if str(j) != 'nan':
        lats4.append(j)
        tilts4_alt.append(tilts4[i])
        lengths4_alt.append(lengths4[i])

lats4_deg = []
for i,j in enumerate(lats4):
    x,y = sunpy.wcs.convert_hpc_hg(0,j)
    lats4_deg.append(y)

### creates north and south tilt lists
tilts4_n = []
tilts4_s = []
lengths4_n = []
lengths4_s = []

abs_lats4_deg = []
for i,j in enumerate(lats4_deg):
    abs_lats4_deg.append(abs(j))
    if j>0: ### northern hemisphere
        tilts4_n.append(tilts4[i])
        lengths4_n.append(lengths4[i])
    else:
        tilts4_s.append(tilts4[i])
        lengths4_s.append(lengths4[i])
        
abs_tilts4_n = []
abs_tilts4_s = []


for i,j in enumerate(tilts4_n):
    abs_tilts4_n.append(abs(j))
    
for i,j in enumerate(tilts4_s):
    abs_tilts4_s.append(abs(j))

 #%%
###############################################################################
################## Histograms, CDFs, Anderson Darling, Etc. ###################
###############################################################################


#ax0,ax1,ax2,ax3 = ax.flatten()

hist_bins = 25

### Length histograms: 123 vs 4
fig, ax0 = plt.subplots()
ax0.hist(lengths123, bins=hist_bins, color='blue', alpha = 0.5)
ax0.hist(lengths4, bins=hist_bins, color='green', alpha = 0.5)
ax0.set_title('Filament Length Histogram')
ax0.set_xlabel('Length (cm)')
#ax0.set_ylabel('Total #')
ax0.set_ylim([0,175])
blue_patch = Patch(color='blue', alpha = 0.5, label='Categories 1, 2, and 3')
green_patch = Patch(color='green', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[blue_patch,green_patch])
fig.savefig("Length_Hist_Unnormed_123v4mod.png",bbox_pad=.1,bbox_inches='tight', label='Category 4')
plt.show()

### Length CDFs: 123 vs 4, normalized
fig, ax1 = plt.subplots()
N123 = len(lengths123)
N4 = len(lengths4)
plt.plot(np.sort(lengths123), np.array(range(N123))/float(N123), color='blue', label='Categories 1, 2, and 3')
plt.plot(np.sort(lengths4), np.array(range(N4))/float(N4), color='green', label='Category 4')
ax1.set_title('Filament Length CDF')
ax1.set_xlabel('Length (cm)')
ax1.set_ylim([0,1.35])
blue_patch = Patch(color='blue', alpha = 0.5, label='Categories 1, 2, and 3')
green_patch = Patch(color='green', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[blue_patch,green_patch])
fig.savefig("Length_CDF_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

stat,crit,sig = stats.anderson_ksamp([lengths123,lengths4])
print stat
print sig


hist_bins = 20

### Tilt Histogram 123 v 4mod, NORTH and SOUTH

fig, ax2 = plt.subplots()
ax2.hist(tilts123_n, bins=hist_bins, color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
ax2.hist(tilts4_n, bins=hist_bins, color='purple', alpha = 0.5, label='Category 4')
ax2.set_title('Filament Tilts: Northern Hemisphere')
ax2.set_xlabel('Tilt Angle (degrees)')
#ax2.set_ylabel('Total #')
ax2.set_ylim([0,27])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Tilt_Hist_North_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

fig, ax2 = plt.subplots()
ax2.hist(tilts123_s, bins=hist_bins, color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
ax2.hist(tilts4_s, bins=hist_bins, color='purple', alpha = 0.5, label='Category 4')
ax2.set_title('Filament Tilts: Southern Hemisphere')
ax2.set_xlabel('Tilt Angle (degrees)')
#ax2.set_ylabel('Total #')
ax2.set_ylim([0,33])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Tilt_Hist_South_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()


hist_bins = 25

### Tilt histograms: 123 vs 4
fig, ax2 = plt.subplots()
ax2.hist(tilts123, bins=hist_bins, color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
ax2.hist(tilts4, bins=hist_bins, color='purple', alpha = 0.5, label='Category 4')
ax2.set_title('Filament Tilt Histogram')
ax2.set_xlabel('Tilt Angle (degrees)')
#ax2.set_ylabel('Total #')
ax2.set_ylim([0,42])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Tilt_Hist_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

### Tilt CDFs: 123 vs 4, normalized
fig, ax3 = plt.subplots()
N123 = len(tilts123)
N4 = len(tilts4)
plt.plot(np.sort(tilts123), np.array(range(N123))/float(N123), color='darkblue', label='Categories 1, 2, and 3')
plt.plot(np.sort(tilts4), np.array(range(N4))/float(N4), color='purple', label='Category 4')
ax3.set_title('Filament Tilt CDF')
ax3.set_xlabel('Tilt Angle (degrees)')
ax3.set_ylim([0,1.35])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Tilt_CDF_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

stat,crit,sig = stats.anderson_ksamp([tilts123,tilts4])
print stat
print sig

### Median latitude histograms: 123 vs 4
fig, ax4 = plt.subplots()
ax4.hist(lats123_deg, bins=hist_bins, color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
ax4.hist(lats4_deg, bins=hist_bins, color='purple', alpha = 0.5, label='Category 4')
ax4.set_title('Filament Latitude Histogram')
ax4.set_xlabel('Latitude (degrees)')
ax4.set_ylim([0,100])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Lat_Hist_Unnormed_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

### Median latitude CDFs: 123 vs 4, normalized
fig, ax5 = plt.subplots()
N123 = len(lats123_deg)
N4 = len(lats4_deg)
plt.plot(np.sort(lats123_deg), np.array(range(N123))/float(N123), color='darkblue', label='Categories 1, 2, and 3')
plt.plot(np.sort(lats4_deg), np.array(range(N4))/float(N4), color='purple', label='Category 4')
ax5.set_title('Filament Latitude CDF')
ax5.set_xlabel('Latitude (degrees)')
ax5.set_ylim([0,1.35])
darkblue_patch = Patch(color='darkblue', alpha = 0.5, label='Categories 1, 2, and 3')
purple_patch = Patch(color='purple', alpha = 0.5, label='Category 4')
plt.legend(frameon=False, handles=[darkblue_patch,purple_patch])
fig.savefig("Lat_CDF_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

stat,crit,sig = stats.anderson_ksamp([lats123,lats4])
print stat
print sig

#fig.tight_layout()
#plt.show()

#%%

###############################################################################
############################# Scatter Plots (Abs) #############################
############################################################################### 
 
### Cat 1 Lat vs. length
corr,pval = stats.pearsonr(abs_lats1, lengths1_alt)
print corr
#fig, ax0 = plt.subplots()
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(left  = 0, right = 2, bottom = 0, top = 3, wspace = 0.2, hspace = 0.3)
                    
ax[0,0].scatter(abs_lats1, lengths1_alt)
ax[0,0].set_xlabel('Median Latitude (degrees)')
ax[0,0].set_ylabel('Median Length (cm)')
ax[0,0].set_title('Latitude vs. Length (Category 1)')
ax[0,0].set_xlim([0,90])
ax[0,0].set_ylim([0,3.0e10])
#fig.savefig("AbsLat_Length1.png",bbox_pad=.1,bbox_inches='tight')

### Cat 2 Lat vs. length
corr,pval = stats.pearsonr(abs_lats2, lengths2_alt)
print corr
#fig, ax1 = plt.subplots()
ax[0,1].scatter(abs_lats2, lengths2_alt)
ax[0,1].set_xlabel('Median Latitude (degrees)')
ax[0,1].set_ylabel('Median Length (cm)')
ax[0,1].set_title('Latitude vs. Length (Category 2)')
ax[0,1].set_xlim([0,90])
ax[0,1].set_ylim([0,3.0e10])
#fig.savefig("AbsLat_Length2.png",bbox_pad=.1,bbox_inches='tight')

### Cat 3 Lat vs. length
corr,pval = stats.pearsonr(abs_lats3, lengths3_alt)
print corr
#fig, ax2 = plt.subplots()
ax[1,0].scatter(abs_lats3, lengths3_alt)
ax[1,0].set_xlabel('Median Latitude (degrees)')
ax[1,0].set_ylabel('Median Length (cm)')
ax[1,0].set_title('Latitude vs. Length (Category 3)')
ax[1,0].set_xlim([0,90])
ax[1,0].set_ylim([0,3.0e10])
#fig.savefig("AbsLat_Length3.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median length (123) vs. median latitude (123)
corr,pval = stats.pearsonr(abs_lats123_deg, lengths123_alt)
print corr
#fig, ax3 = plt.subplots()
ax[1,1].scatter(abs_lats123_deg, lengths123_alt)
ax[1,1].set_xlabel('Median Latitude (degrees)')
ax[1,1].set_ylabel('Median Length (cm)')
ax[1,1].set_title('Latitude vs. Length (123)')
ax[1,1].set_xlim([0,90])
ax[1,1].set_ylim([0,3.0e10])
#fig.savefig("AbsLat_Length123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median lengths (4) vs. median latitude (4)
corr,pval = stats.pearsonr(abs_lats4_deg, lengths4_alt)
print corr
#fig, ax4 = plt.subplots()
ax[2,0].scatter(abs_lats4_deg, lengths4_alt)
ax[2,0].set_xlabel('Median Latitude (degrees)')
ax[2,0].set_ylabel('Median Length (cm)')
ax[2,0].set_title('Latitude vs. Length (4)')
ax[2,0].set_xlim([0,90])
ax[2,0].set_ylim([0,3.0e10])
#fig.savefig("AbsLat_Length4.png",bbox_pad=.1,bbox_inches='tight')

ax[2,1].set_axis_off()

fig.savefig("AbsLat_Length_123v4mod.png",bbox_pad=.1,bbox_inches='tight')


### Cat 1 Lat vs. tilt

fig, ax = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(left  = 0, right = 2, bottom = 0, top = 3, wspace = 0.2, hspace = 0.3)

corr,pval = stats.pearsonr(abs_lats1, tilts1_alt)
print corr
#fig, ax = plt.subplots()
ax[0,0].scatter(abs_lats1, tilts1_alt)
ax[0,0].set_xlabel('Median Latitude (degrees)')
ax[0,0].set_ylabel('Median Tilt Angle (degrees)')
ax[0,0].set_title('Latitude vs. Tilt (Category 1)')
ax[0,0].set_xlim([0,90])
ax[0,0].set_ylim([-100,100])
#fig.savefig("AbsLat_Tilt1.png",bbox_pad=.1,bbox_inches='tight')

### Cat 2 Lat vs. tilt
corr,pval = stats.pearsonr(abs_lats2, tilts2_alt)
print corr
#fig, ax = plt.subplots()
ax[1,0].scatter(abs_lats2, tilts2_alt)
ax[1,0].set_xlabel('Median Latitude (degrees)')
ax[1,0].set_ylabel('Median Tilt Angle (degrees)')
ax[1,0].set_title('Latitude vs. Tilt (Category 2)')
ax[1,0].set_xlim([0,90])
ax[1,0].set_ylim([-100,100])
#fig.savefig("AbsLat_Tilt2.png",bbox_pad=.1,bbox_inches='tight')

### Cat 3 Lat vs. tilt
corr,pval = stats.pearsonr(abs_lats3, tilts3_alt)
print corr
#fig, ax = plt.subplots()
ax[0,1].scatter(abs_lats3, tilts3_alt)
ax[0,1].set_xlabel('Median Latitude (degrees)')
ax[0,1].set_ylabel('Median Tilt Angle (degrees)')
ax[0,1].set_title('Latitude vs. Tilt (Category 3)')
ax[0,1].set_xlim([0,90])
ax[0,1].set_ylim([-100,100])
#fig.savefig("AbsLat_Tilt3.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (123) vs. median latitude (123)
corr,pval = stats.pearsonr(abs_lats123_deg, tilts123_alt)
print corr
#fig, ax = plt.subplots()
ax[1,1].scatter(abs_lats123_deg, tilts123_alt)
ax[1,1].set_xlabel('Median Latitude (degrees)')
ax[1,1].set_ylabel('Median Tilt Angle (degrees)')
ax[1,1].set_title('Latitude vs. Tilt (123)')
ax[1,1].set_xlim([0,90])
ax[1,1].set_ylim([-100,100])
#fig.savefig("AbsLat_Tilt123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (4) vs. median latitude (4)
corr,pval = stats.pearsonr(abs_lats4_deg, tilts4_alt)
print corr
#fig, ax = plt.subplots()
ax[2,0].scatter(abs_lats4_deg, tilts4_alt)
ax[2,0].set_xlabel('Median Latitude (degrees)')
ax[2,0].set_ylabel('Median Tilt Angle (degrees)')
ax[2,0].set_title('Latitude vs. Tilt (4)')
ax[2,0].set_xlim([0,90])
ax[2,0].set_ylim([-100,100])
#fig.savefig("AbsLat_Tilt4.png",bbox_pad=.1,bbox_inches='tight')

ax[2,1].set_axis_off()

fig.savefig("AbsLat_Tilt_123v4mod.png",bbox_pad=.1,bbox_inches='tight')


###############################################################################
########################### Scatter Plots (No Abs) ############################
############################################################################### 
 
### Cat 1 Lat vs. length
 
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(left  = 0, right = 2, bottom = 0, top = 3, wspace = 0.2, hspace = 0.3) 
 
corr,pval = stats.pearsonr(lats1_deg, lengths1_alt)
print corr
#fig, ax = plt.subplots()
ax[0,0].scatter(lats1_deg, lengths1_alt)
ax[0,0].set_xlabel('Median Latitude (degrees)')
ax[0,0].set_ylabel('Median Length (cm)')
ax[0,0].set_title('Latitude vs. Length (Category 1)')
ax[0,0].set_xlim([-90,90])
ax[0,0].set_ylim([0,3.0e10])
#fig.savefig("Lat_Length1.png",bbox_pad=.1,bbox_inches='tight')

### Cat 2 Lat vs. length
corr,pval = stats.pearsonr(lats2_deg, lengths2_alt)
print corr
#fig, ax = plt.subplots()
ax[0,1].scatter(lats2_deg, lengths2_alt)
ax[0,1].set_xlabel('Median Latitude (degrees)')
ax[0,1].set_ylabel('Median Length (cm)')
ax[0,1].set_title('Latitude vs. Length (Category 2)')
ax[0,1].set_xlim([-90,90])
ax[0,1].set_ylim([0,3.0e10])
#fig.savefig("Lat_Length2.png",bbox_pad=.1,bbox_inches='tight')

### Cat 3 Lat vs. length
corr,pval = stats.pearsonr(lats3_deg, lengths3_alt)
print corr
#fig, ax = plt.subplots()
ax[1,0].scatter(lats3_deg, lengths3_alt)
ax[1,0].set_xlabel('Median Latitude (degrees)')
ax[1,0].set_ylabel('Median Length (cm)')
ax[1,0].set_title('Latitude vs. Length (Category 3)')
ax[1,0].set_xlim([-90,90])
ax[1,0].set_ylim([0,3.0e10])
#fig.savefig("Lat_Length3.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median length (123) vs. median latitude (123)
corr,pval = stats.pearsonr(lats123_deg, lengths123_alt)
print corr
#fig, ax = plt.subplots()
ax[1,1].scatter(lats123_deg, lengths123_alt)
ax[1,1].set_xlabel('Median Latitude (degrees)')
ax[1,1].set_ylabel('Median Length (cm)')
ax[1,1].set_title('Latitude vs. Length (123)')
ax[1,1].set_xlim([-90,90])
ax[1,1].set_ylim([0,3.0e10])
#fig.savefig("Lat_Length123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median lengths (4) vs. median latitude (4)
corr,pval = stats.pearsonr(lats4_deg, lengths4_alt)
print corr
#fig, ax = plt.subplots()
ax[2,0].scatter(lats4_deg, lengths4_alt)
ax[2,0].set_xlabel('Median Latitude (degrees)')
ax[2,0].set_ylabel('Median Length (cm)')
ax[2,0].set_title('Latitude vs. Length (4)')
ax[2,0].set_xlim([-90,90])
ax[2,0].set_ylim([0,3.0e10])
#fig.savefig("Lat_Length4.png",bbox_pad=.1,bbox_inches='tight')

ax[2,1].set_axis_off()

fig.savefig("Lat_Length_123v4mod.png",bbox_pad=.1,bbox_inches='tight')

fig, ax = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(left  = 0, right = 2, bottom = 0, top = 3, wspace = 0.2, hspace = 0.3) 

### Cat 1 Lat vs. tilt
corr,pval = stats.pearsonr(lats1_deg, tilts1_alt)
print corr

med_tilt_n = np.median(tilts1_n)
med_tilt_s = np.median(tilts1_s)

#fig, ax = plt.subplots()
ax[0,0].scatter(lats1_deg, tilts1_alt, color='k', marker ='.')
ax[0,0].plot([-90,90],[med_tilt_n, med_tilt_n], 'p-')
ax[0,0].plot([-90,90],[med_tilt_s, med_tilt_s], 'g-')
ax[0,0].set_xlabel('Median Latitude (degrees)')
ax[0,0].set_ylabel('Median Tilt Angle (degrees)')
ax[0,0].set_title('Latitude vs. Tilt (Category 1)')
ax[0,0].set_xlim([-90,90])
ax[0,0].set_ylim([-100,100])
#fig.savefig("Lat_Tilt1.png",bbox_pad=.1,bbox_inches='tight')

### Cat 2 Lat vs. tilt
corr,pval = stats.pearsonr(lats2_deg, tilts2_alt)
print corr

med_tilt_n = np.median(tilts2_n)
med_tilt_s = np.median(tilts2_s)

#fig, ax = plt.subplots()
ax[0,1].scatter(lats2_deg, tilts2_alt, color='k', marker ='.')
ax[0,1].plot([-90,90],[med_tilt_n, med_tilt_n], 'p-')
ax[0,1].plot([-90,90],[med_tilt_s, med_tilt_s], 'g-')
ax[0,1].set_xlabel('Median Latitude (degrees)')
ax[0,1].set_ylabel('Median Tilt Angle (degrees)')
ax[0,1].set_title('Latitude vs. Tilt (Category 2)')
ax[0,1].set_xlim([-90,90])
ax[0,1].set_ylim([-100,100])
#fig.savefig("Lat_Tilt2.png",bbox_pad=.1,bbox_inches='tight')

### Cat 3 Lat vs. tilt
corr,pval = stats.pearsonr(lats3_deg, tilts3_alt)
print corr

med_tilt_n = np.median(tilts3_n)
med_tilt_s = np.median(tilts3_s)

#fig, ax = plt.subplots()
ax[1,0].scatter(lats3_deg, tilts3_alt, color='k', marker ='.')
ax[1,0].plot([-90,90],[med_tilt_n, med_tilt_n], 'p-')
ax[1,0].plot([-90,90],[med_tilt_s, med_tilt_s], 'g-')
ax[1,0].set_xlabel('Median Latitude (degrees)')
ax[1,0].set_ylabel('Median Tilt Angle (degrees)')
ax[1,0].set_title('Latitude vs. Tilt (Category 3)')
ax[1,0].set_xlim([-90,90])
ax[1,0].set_ylim([-100,100])
#fig.savefig("Lat_Tilt3.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (123) vs. median latitude (123)
corr,pval = stats.pearsonr(lats123_deg, tilts123_alt)
print corr

med_tilt_n = np.median(tilts123_n)
med_tilt_s = np.median(tilts123_s)

#fig, ax = plt.subplots()
ax[1,1].scatter(lats123_deg, tilts123_alt, color='k', marker ='.')
ax[1,1].plot([-90,90],[med_tilt_n, med_tilt_n], 'p-')
ax[1,1].plot([-90,90],[med_tilt_s, med_tilt_s], 'g-')
ax[1,1].set_xlabel('Median Latitude (degrees)')
ax[1,1].set_ylabel('Median Tilt Angle (degrees)')
ax[1,1].set_title('Latitude vs. Tilt (123)')
ax[1,1].set_xlim([-90,90])
ax[1,1].set_ylim([-100,100])
#fig.savefig("Lat_Tilt123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (4) vs. median latitude (4)
corr,pval = stats.pearsonr(lats4_deg, tilts4_alt)
print corr

med_tilt_n = np.median(tilts4_n)
med_tilt_s = np.median(tilts4_s)

#fig, ax = plt.subplots()
ax[2,0].scatter(lats4_deg, tilts4_alt, color='k', marker ='.')
ax[2,0].plot([-90,90],[med_tilt_n, med_tilt_n], 'p-')
ax[2,0].plot([-90,90],[med_tilt_s, med_tilt_s], 'g-')
ax[2,0].set_xlabel('Median Latitude (degrees)')
ax[2,0].set_ylabel('Median Tilt Angle (degrees)')
ax[2,0].set_title('Latitude vs. Tilt (4)')
ax[2,0].set_xlim([-90,90])
ax[2,0].set_ylim([-100,100])
#fig.savefig("Lat_Tilt4.png",bbox_pad=.1,bbox_inches='tight')

ax[2,1].set_axis_off()

fig.savefig("Lat_Tilt_lines_123v4mod.png",bbox_pad=.1,bbox_inches='tight')







### Scatter plot of median tilts (123) vs. median length (123)
fig, ax = plt.subplots()
ax.plot(tilts123_n, lengths123_n,'bo')
ax.plot(tilts123_s, lengths123_s,'ro')
ax.set_xlabel('Median Tilt Angle (degrees)')
ax.set_ylabel('Median Length (cm)')
ax.set_title('Tilt vs. Length (Categories 1, 2, and 3)')
fig.savefig("Tilt_Length_Scatter_ns123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (4) vs. median length (4)
fig, ax = plt.subplots()
ax.plot(tilts4_n, lengths4_n,'bo')
ax.plot(tilts4_s, lengths4_s,'ro')
ax.set_xlabel('Median Tilt Angle (degrees)')
ax.set_ylabel('Median Length (cm)')
ax.set_title('Tilt vs. Length (Category 4)')
fig.savefig("Tilt_Length_Scatter_ns4mod.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (123) vs. median length (123)
fig, ax = plt.subplots()
ax.plot(tilts123_n, lengths123_n,'bo')
ax.plot(tilts123_s, lengths123_s,'ro')
ax.set_xlabel('Median Tilt Angle (degrees)')
ax.set_ylabel('Median Length (cm)')
ax.set_title('Absolute Value of Tilt vs. Length (Categories 1, 2, and 3)')
fig.savefig("Abs_Tilt_Length_Scatter_ns123.png",bbox_pad=.1,bbox_inches='tight')

### Scatter plot of median tilts (4) vs. median length (4)
fig, ax = plt.subplots()
ax.plot(abs_tilts4_n, lengths4_n,'bo')
ax.plot(abs_tilts4_s, lengths4_s,'ro')
ax.set_xlabel('Median Tilt Angle (degrees)')
ax.set_ylabel('Median Length (cm)')
ax.set_title('Absolute Value of Tilt vs. Length (4)')
fig.savefig("Abs_Tilt_Length_Scatter_ns4mod.png",bbox_pad=.1,bbox_inches='tight')

#fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2, sharex='col', sharey='row')

#%%
###############################################################################
####################### Logistic Regression - Length ##########################
############################################################################### 
"""
#sm.Logit()

hist_bins = 30

### Stacked Length histograms: 123 vs 4 for logistic regression
fig, ax0 = plt.subplots()
ax0.hist(lengths123, bins=hist_bins, color='blue', stacked=True, cumulative=True)
ax0.set_title('Filament Length')
ax0.set_xlabel('Length (cm)')
fig.savefig("Length_LogReg.png",bbox_pad=.1,bbox_inches='tight')
plt.show()
"""

### list of category 1, 2, 3, and 4 lengths combined
lengths1234 = lengths123 + lengths4
### create a list of 1's and 0's: 1 means the length is from cat123, 0 means from cat4
### binary lengths corresponding to lengths1234
bi_lengths = [0]*len(lengths1234)
for i,j in enumerate(bi_lengths):
    if i < len(lengths123):
        bi_lengths[i] = 1

### divide lengths into bins
### starting location of bins
sbins = [0.0, 0.3e10, 0.4e10, 0.5e10, 0.6e10, 0.8e10, 1.1e10, 1.5e10]

### get the maximum length
maxl = max(lengths1234)
### fraction with filaments in each bin
fbin = []
### center of bin
cbin =[]

### loop over all bins
for i,sbin in enumerate(sbins):
    ### go all way to end if on the last bin
    if i == len(sbins)-1: 
        ebin = maxl+1.0
    else:
        ebin = sbins[i+1]
    ### get where x values are in range and have cavities
    filwcav, = np.where((np.asarray(lengths123) >= sbin) & (np.asarray(lengths123) < ebin))
    ### get total number of cavities in bin
    totacav, = np.where((np.asarray(lengths1234) >= sbin) & (np.asarray(lengths1234) < ebin))
    ### store values for later
    fbin.append(float(filwcav.size)/float(totacav.size))
    print totacav.size
    cbin.append(sbin + (ebin-sbin)/2.)

"""
### plot these point estimates for logistic regression
fig, ax = plt.subplots()
plt.plot(cbin, fbin, 'bo')
ax.set_xlim([0,2.5e10])
ax.set_ylim([0,1])
ax.set_title('Filament Length')
ax.set_xlabel('Length (cm)')
fig.savefig("Length_Logit_Points.png",bbox_pad=.1,bbox_inches='tight')
plt.show()
"""

### create empty data frame
cav_df = pd.DataFrame()
"""
### finds the best guess for the intercept based on above binning
for i,j in enumerate(fbin):
    close_to_50 = []
    close_to_50.append(abs(j - 0.5))
closest = min(close_to_50)
closest_ind = np.where(np.asarray(fbin) == closest + 0.5)
# Your intercept guess
inter_guess = cbin[int(list(closest_ind)[0])]
"""

### Intercept guess
inter_guess = cbin[4]
print inter_guess

### store list in data frame
cav_df['bi_lengths'] = bi_lengths
cav_df['lengths1234'] = lengths1234
### create an array of constants that hold a place for the intercept 
cav_df['intercept_l'] = [1]*len(lengths1234) 

### tell the logit function what column to use
use_cols = cav_df.columns[1:]

### logistic regression for lengths
logit = sm.Logit(cav_df['bi_lengths'],cav_df[use_cols])
length_result = logit.fit()
print length_result.summary()


mono_lengths = np.linspace(min(lengths1234), max(lengths1234), len(cav_df['bi_lengths']), endpoint=True)
cav_df['mono_lengths'] = mono_lengths
len_predict = length_result.predict(cav_df[['mono_lengths','intercept_l']])

### plot these point estimates for logistic regression
fig, ax = plt.subplots()
ax.plot(lengths1234, bi_lengths, 'bo')
ax.plot(mono_lengths, len_predict, 'r-',)
ax.set_xlim([0,2.75e10])
ax.set_ylim([-0.02, 1.02])
ax.set_title('Filament Length')
ax.set_xlabel('Length (cm)')
ax.set_ylabel('Proportion of Filaments with Cavities')
fig.savefig("LogReg_Length_Bi_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

#%%
###############################################################################
####################### Logistic Regression - Tilt ############################
############################################################################### 

### list of category 1, 2, 3, and 4 tilts combined
tilts1234 = tilts123 + tilts4
### create a list of 1's and 0's: 1 means the tilt is from cat123, 0 means from cat4
### binary tilts corresponding to tilts1234
bi_tilts = [0]*len(tilts1234)
for i,j in enumerate(bi_tilts):
    if i < len(tilts123):
        bi_tilts[i] = 1

### divide tilts into bins
### starting location of bins
sbins = [-90, -45, -30, -20, -10, 0, 10, 20, 30, 45]

### fraction with filaments in each bin
fbin = []
### center of bin
cbin =[]

### loop over all bins
for i,sbin in enumerate(sbins):
    ### go all way to end if on the last bin
    if i == len(sbins)-1: 
        ebin = 90
    else:
        ebin = sbins[i+1]
    ### get where x values are in range and have cavities
    filwcav, = np.where((np.asarray(tilts123) >= sbin) & (np.asarray(tilts123) < ebin))
    ### get total number of cavities in bin
    totacav, = np.where((np.asarray(tilts1234) >= sbin) & (np.asarray(tilts1234) < ebin))
    ### store values for later
    fbin.append(float(filwcav.size)/float(totacav.size))
    print totacav.size
    cbin.append(sbin + (ebin-sbin)/2.)

### Intercept guess
inter_guess = cbin[7]
print inter_guess

### store list in data frame
cav_df['bi_tilts'] = bi_tilts
cav_df['tilts1234'] = tilts1234
### create an array of constants that hold a place for the intercept 
cav_df['intercept_t'] = [1]*len(tilts1234) 

### tell the logit function what column to use
use_cols = cav_df.columns[5:]

### logistic regression for tilts
logit = sm.Logit(cav_df['bi_tilts'],cav_df[use_cols])
tilt_result = logit.fit()
print tilt_result.summary()

mono_tilts = np.linspace(min(tilts1234), max(tilts1234), len(cav_df['bi_lengths']), endpoint=True)
cav_df['mono_tilts'] = mono_tilts
tilt_predict = tilt_result.predict(cav_df[['mono_tilts','intercept_t']])


### plot these point estimates for logistic regression
fig, ax = plt.subplots()
ax.plot(tilts1234, bi_tilts, 'bo')
ax.plot(mono_tilts, tilt_predict, 'r-',)
ax.set_xlim([-90,90])
ax.set_ylim([-0.02,1.02])
ax.set_title('Filament Tilt')
ax.set_xlabel('Tilt (degrees)')
ax.set_ylabel('Proportion of Filaments with Cavities')
fig.savefig("LogReg_Tilt_Bi_123v4mod.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

"""
### Stacked Tilt histograms: 123 vs 4 for logistic regression
fig, ax2 = plt.subplots()
#ax2.hist(tilts123+tilts4, bins=hist_bins, color='darkblue', stacked=True, cumulative=True)
ax2.hist(tilts123, bins=hist_bins, color='lightgreen', stacked=True, cumulative=True)
ax2.set_title('Filament Tilt Histogram')
ax2.set_xlabel('Tilt Angle (degrees)')
darkblue_patch = Patch(color='darkblue', label='All Categories')
lightgreen_patch = Patch(color='lightgreen', label='Categories 123 Only')
#plt.legend(frameon=False, handles=[darkblue_patch,lightgreen_patch])
fig.savefig("Tilt_LogReg.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

#sm.Logit(lengths123,lengths4)




### Stacked Median latitude histograms: 123 vs 4 for Logistic Regression
fig, ax4 = plt.subplots()
#ax4.hist(lats123_deg+lats4_deg, bins=hist_bins, color='purple', stacked=True, cumulative=True)
ax4.hist(lats123_deg, bins=hist_bins, color='orange', stacked=True, cumulative=True)
ax4.set_title('Filament Latitude Histogram')
ax4.set_xlabel('Latitude (degrees)')
purple_patch = Patch(color='purple', label='All Categories')
orange_patch = Patch(color='orange', label='Categories 123 Only')
#plt.legend(frameon=False, handles=[purple_patch,orange_patch])
fig.savefig("Lat_LogReg.png",bbox_pad=.1,bbox_inches='tight')
plt.show()

"""
########################### REFERENCE FOR LATER ###############################
"""
test_track, = np.where(fil['track_id'] == cat1['C212064209N'][0])
print test_track
fil['track_id'][test_track]
fil['meanx'][test_track]
fil['event_starttime_dt'][test_track]

### counts numbers of items in a dictionary
sum(map(len,cat1.values()))
"""