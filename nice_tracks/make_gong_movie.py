from datetime import datetime, timedelta
import numpy as np
import grab_gong
import add_properties as ap
import pandas as pd
from multiprocessing import Pool
from create_plots import halpha_plot
import os
def get_best_track(tid):

    track, = np.where(dat['track_id'].values == tid)
   
    stid = '{0:6.0f}'.format(tid).replace(' ','0')
    spdir = '{0}{1}/'.format(pdir,stid)
#create plot subdirectory based on plot id
    try:
        os.mkdir(spdir)
    except:
        print 'Directory {0} exists. Proceeding..'.format(stid)

    
        
    
#get the start and end time of the track    
    start = dat['event_starttime_dt'][track[0]]
    end   = dat['event_endtime_dt'][track[-1]]
    #make sure files exist locally and output filenames
    flist = grab_gong.main(start,end).filelist
    
    for i in flist:
        out = halpha_plot(dat,i,tid,spdir)
        out.plot_filament_track()
    


#for inital testing
#fmt = '%Y/%m/%dT%H:%M:%S'
#perhaps I should make year long halpha movies
#start = datetime.strptime('2013/01/00T00:00:00',fmt)
#end = datetime.strptime('2013/01/31T23:59:59',fmt)

infile = '../init_data/FITracked_3yr.txt'
#dat = ascii.read('../init_data/FITracked_3yr.txt',delimiter='\t',guess=False)
dat = pd.read_csv(infile,delimiter='\t')
dfmt = '%Y-%m-%dT%H:%M:%S'

#add variables like datetime and average position to dat
dat = ap.add_props(dat).dat

#set up plot directory
sdir = os.getcwd()
pdir = sdir+'/track_plots/'

goodtracks = np.loadtxt('list_of_excellent_tracks')

nproc = 8
#do in parallel all good tracks
pool =Pool(processes=nproc)
out = pool.map(get_best_track,goodtracks)
pool.close()

#get_best_track(goodtracks[0])



