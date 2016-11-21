from datetime import datetime, timedelta
import numpy as np
import grab_gong
import add_properties as ap
import pandas as pd
from multiprocessing import Pool
from create_plots import halpha_plot
import os
def get_best_track(start,end):
    #make sure files exist locally and output filenames
    flist = grab_gong.main(start,end).filelist
    return flist
    
def create_images(i):
    out = halpha_plot(dat,i,pdir)
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

try:
    os.mkdir(pdir)
except:
    print 'Directory {0} already exists'.format(pdir)


nproc = 8
#do in parallel all good tracks
#pool =Pool(processes=nproc)
#out = pool.map(get_best_track,goodtracks)
#pool.close()


start = '2012-01-01T00:00:00'
end   = '2015-12-31T00:00:00'


ostart = datetime.strptime(start,dfmt)
oend   = datetime.strptime(end,dfmt)

flist = get_best_track(ostart,oend)
#do in parallel 
pool =Pool(processes=nproc)
out = pool.map(create_images,flist)
pool.close()



