from datetime import datetime, timedelta
import make_movie
import numpy as np
import grab_gong
import add_properties as ap
import pandas as pd
from multiprocessing import Pool
from create_plots import halpha_plot
import os
def get_best_track(start,end,local=False):
    #make sure files exist locally and output filenames
    flist = grab_gong.main(start,end,local=local).filelist
    return flist
    
def create_images(i):
    out = halpha_plot(dat,i,pdir)
    out.plot_filament_track()
    


#for inital testing
#fmt = '%Y/%m/%dT%H:%M:%S'
#perhaps I should make year long halpha movies
#start = datetime.strptime('2013/01/00T00:00:00',fmt)
#end = datetime.strptime('2013/01/31T23:59:59',fmt)

infile = '../init_data/FITracked_3yr.pic'
infile = 'concatentated_per_shape_3yr_file.pic'

pickled = os.path.isfile(infile)
if pickled:#use pickle file if it already exits
    dat = pd.read_pickle(infile)
else: #create pickle file if doesnt exist
    infile = '../init_data/FITracked_3yr.txt'
    #dat = ascii.read('../init_data/FITracked_3yr.txt',delimiter='\t',guess=False)
    dat = pd.read_csv(infile,delimiter='\t')
    #add variables like datetime and average position to dat
    dat = ap.add_props(dat).dat
    dat.to_pickle('../init_data/FITracked_3yr.pic')

dfmt = '%Y-%m-%dT%H:%M:%S'
#set up plot directory
sdir = os.getcwd()
pdir = sdir+'/track_plots/'

try:
    os.mkdir(pdir)
except:
    print 'Directory {0} already exists'.format(pdir)
try:
    os.mkdir(pdir+'symlinks')
except:
    print 'Directory {0} already exists'.format(pdir+'symlinks')


nproc = 8
#do in parallel all good tracks
#pool =Pool(processes=nproc)
#out = pool.map(get_best_track,goodtracks)
#pool.close()


start = '2012-01-01T00:00:00'
end   = '2014-12-01T00:00:00'


ostart = datetime.strptime(start,dfmt)
oend   = datetime.strptime(end,dfmt)

rdir = os.getcwd() #return to current working directory before making movie

print 'STARTING DOWNLOAD'
flist = get_best_track(ostart,oend,local=True)

os.chdir(rdir)
#do in parallel 
#for mm in flist: create_images(mm)
pool =Pool(processes=nproc)
out = pool.map(create_images,flist)
pool.close()


os.chdir(rdir)
#create movie from image files
imov = make_movie.create_movie(w0=2048,h0=2048,nproc=4,outmov='halpha_filament_movie_per_shape.mp4')
imov.create_movie()

