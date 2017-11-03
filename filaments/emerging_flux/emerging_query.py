from sunpy.net import hek
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import os

def format_string(st):

    out = st.replace('/','').replace(':','').replace(' ','_')
    return out

#retrieve desired cadence from file list
def des_cad(start,end,delta):
    """Create an array from start to end with desired cadence"""
    curr = start
    while curr < end:
        yield curr
        curr += delta




tstart = '2011/12/31 00:00:00'
tend   = '2015/01/01 00:00:00'
dfmt = '%Y/%m/%d %H:%M:%S'
tstart = datetime(2012,1,1,0,0,0)
tend   = datetime(2014,11,30,0,0,0)
#tend   = datetime(2012,1,3,0,0,0)

cad = 2.*60.*60.*24.#seconds in a day


#get query cadence
real_cad = [result for result in des_cad(tstart,tend,timedelta(seconds=cad))]


#create hek query class
client = hek.HEKClient()

#filament eruptions
event_type = 'EF'


#query the HEK


t = 0
for k,m in enumerate(real_cad):

    #output file name
    fname = 'query_output/{0}_ef.pic'.format(m.strftime('%Y%m%d'))
    
    #check to see if file already exists
    testfile = os.path.isfile(fname)
   
    #if it does just restore and continue
    if testfile:
        ef_df = pd.read_pickle(fname)
        keys = ef_df.columns
        continue
    else:
        failed = True
        #get filament eruption in time rangeh
        while failed:
            try:
                result = client.search(hek.attrs.Time(m,m+timedelta(seconds=cad)),hek.attrs.EventType(event_type))
                failed = False
            except:
                failed = False

        #get the keys to store for emerging flux
        if k == 0:
            #return possible key 
            keys = result[0].keys()
            
            #create pandas data frame from keys
            ef_df = pd.DataFrame(columns=keys)
        
        #turn dictionary into pandas Data frame
        for j,i in enumerate(result): 
            ef_df.loc[t] = [i[k] for k in keys]
            t+= 1

        #save current table to file
        ef_df.to_pickle(fname)
            


#change index to input time
ef_df.set_index(pd.to_datetime(ef_df['event_starttime']),inplace=True)
ef_df.to_pickle('query_output/all_ef_{0}-{1}.pic'.format(tstart.strftime('%Y%m%d'),tend.strftime('%Y%m%d')))

##Levels of flares
#f_l = ['X','M','C','B']
#
##Total number of flares
#f_df['T'] = 0
#
##make columns for is flare classification 
#for i in f_l: 
#    f_df[i] = 0
#    #set values to 1 where first character matches value
#    f_df[i][f_df.goes.str[0] == i] = 1
#    f_df['T'][f_df.goes.str[0] == i] = 1
#
##day resampling
#d_df = f_df.resample('1D').sum()
#
#
##noaa number resampling 
#n_df = f_df.groupby(['AR']).sum().reset_index()
#n_df.set_index(n_df['AR'],inplace=True) # = f_df.groupby(['AR']).sum().reset_index()
#
#
#
#
##write output to file
#tout = format_string(tstart)
#out_f = open('flares_'+tout+'.txt','w')
#
#out_f.write('#######DATE BREAKDOWN##############\n')
#out_f.write(d_df[['X','M','C','B','T']].to_string()+'\n')
#out_f.write('#########AR BREAKDOWN##############\n')
#out_f.write(n_df[['X','M','C','B','T']].to_string()+'\n')
##set datetime to be the index
#out_f.close()