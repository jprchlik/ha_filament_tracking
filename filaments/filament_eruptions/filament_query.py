from sunpy.net import hek
import numpy as np
import pandas as pd
from datetime import datetime,timedelta

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
event_type = 'FE'

#full list of filament keys as of 2017/09/29
keys = [u'concept', u'frm_versionnumber', u'gs_movieurl', u'hrc_coord', u'hpc_bbox', u'area_atdiskcenter',
        u'event_mapurl', u'event_c1error', u'obs_dataprepurl', u'hgc_coord', u'frm_identifier', u'intensmean',
        u'boundbox_c2ur', u'event_coordunit', u'gs_thumburl', u'obs_meanwavel', u'bound_ccnsteps', u'outflow_width',
        u'hgs_coord', u'bound_chaincode', u'frm_daterun', u'intensmedian', u'bound_ccstartc1', u'frm_paramset',
        u'bound_ccstartc2', u'event_coord2', u'event_coord3', u'outflow_widthunit', u'event_coord1', u'event_importance',
        u'kb_archivdate', u'event_title', u'hrc_r', u'hgc_bbox', u'skel_chaincode', u'intenstotal',
        u'hrc_a', u'area_atdiskcenteruncert', u'event_probability', u'hrc_boundcc', u'event_description', u'search_frm_name',
        u'eventtype', u'obs_channelid', u'ar_mcintoshcls', u'frm_institute', u'frm_contact', u'ar_noaaclass',
        u'search_observatory', u'boundbox_c1ur', u'hgs_boundcc', u'boundbox_c2ll', u'area_unit', u'intensskew',
        u'hpc_coord', u'frm_name', u'obs_levelnum', u'area_uncert', u'ar_zurichcls', u'active',
        u'search_instrument', u'hpc_radius', u'event_importance_num_ratings', u'obs_includesnrt', u'event_testflag', u'hpc_y',
        u'hpc_x', u'hpc_boundcc', u'event_score', u'obs_lastprocessingdate', u'refs_orig', u'ar_numspots',
        u'intensvar', u'outflow_speed', u'event_avg_rating', u'frm_url', u'ar_compactnesscls', u'comment_count',
        u'event_npixels', u'event_clippedspatial', u'obs_wavelunit', u'frm_humanflag', u'hcr_checked', u'event_expires',
        u'noposition', u'event_peaktime', u'kb_archivist', u'SOL_standard', u'event_coordsys', u'hgc_boundcc',
        u'gs_galleryid', u'event_maskurl', u'outflow_speedunit', u'skel_startc1', u'skel_startc2', u'obs_title',
        u'event_type', u'refs', u'hgc_x', u'hgc_y', u'outflow_length', u'kb_archivid',
        u'ar_penumbracls', u'obs_firstprocessingdate', u'event_endtime', u'hrc_bbox', u'outflow_lengthunit', u'search_channelid',
        u'skel_nsteps', u'rasterscan', u'ar_mtwilsoncls', u'skel_curvature', u'event_pixelunit', u'outflow_transspeed',
        u'ar_noaanum', u'revision', u'hgs_x', u'hgs_y', u'intenskurt', u'ar_polarity',
        u'obs_instrument', u'frm_specificid', u'boundbox_c1ll', u'area_raw', u'rasterscantype', u'sum_overlap_scores',
        u'obs_observatory', u'intensunit', u'chaincodetype', u'outflow_openingangle', u'gs_imageurl', u'hgs_bbox',
        u'event_starttime', u'event_clippedtemporal', u'intensmax', u'intensmin', u'hpc_geom', u'event_c2error']


#create pandas data frame from keys
fe_df = pd.DataFrame(columns=keys)

#query the HEK


t = 0
for m in real_cad:

    failed = True
    #get filament eruption in time rangeh
    while failed:
        try:
            result = client.search(hek.attrs.Time(m,m+timedelta(seconds=cad)),hek.attrs.EventType(event_type),hek.attrs.OBS.Instrument == 'AIA')
            failed = False
        except:
            failed = False
    
    #turn dictionary into pandas Data frame
    for j,i in enumerate(result): 
        fe_df.loc[t] = [i[k] for k in keys]
        t+= 1

    #save current table to file
    fe_df.to_pickle('query_output/{0}_fe.pic'.format(m.strftime('%Y%m%d')))
            


#change index to input time
fe_df.set_index(pd.to_datetime(fe_df['event_starttime']),inplace=True)
fe_df.to_pickle('query_output/all_fe_{0}-{1}.pic'.format(tstart.strftime('%Y%m%d'),tend.strftime('%Y%m%d')))

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