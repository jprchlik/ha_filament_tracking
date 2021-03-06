import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 28
#Set default image dpi
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600

import matplotlib.dates as mdates

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes,InsetPosition

from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fancy_plot import fancy_plot
import pandas as pd
import numpy as np
from datetime import timedelta,datetime
from shapely.wkt  import dumps, loads

import scipy.stats as stats
#import statsmodels.api as sm

#set up distribution in order
def setup_dis(x,col='med_tilt'):
    x.set_index(x['track_id'],inplace=True)
    x.sort_values(by=col,inplace=True)
    x[len(x)] = x.iloc[-1]
    x['dis']  = np.linspace(0.,1.,len(x))

    #add value so all distributions start on Jan. 1st
    x.loc[-1] = [np.nan]*x.shape[1]
    return x


def real_resamp(x,dates,col='med_tilt'):

    #keep only good values 
    x = x[x[col].notnull()]

    #setup binned dataframe
    y = pd.DataFrame(index=dates)
    y.loc[:,col+'_mean'] = np.nan
    y.loc[:,col+'_med'] = np.nan
    y.loc[:,col+'_std'] = np.nan
    y.loc[:,col+'_sum'] = np.nan
    y.loc[:,col+'_cnt'] = np.nan

    #Add extra parameters for emerging flux 2018/06/27 J. Prchlik
    if col == 'area_atdiskcenter':
        y.loc[:,col+'_unc'] = np.nan #Area uncertainty
        y.loc[:,col+'_dtt'] = np.nan #Time in bin in days
    
   
    #total number of dates
    t = len(dates)


    #remove none values
    #x = x[x.astype(str).ne('None').all(1)]

    for j,i in enumerate(dates):

        if j < t-1:
            use, = np.where((x.index >= i) & (x.index < dates[j+1]))
        else:
            use, = np.where(x.index >= i)
          
        if use.size > 0:
            y.loc[i,col+'_mean'] = np.mean(x[col].values[use])
            y.loc[i,col+'_med']  = np.median(x[col].values[use])
            y.loc[i,col+'_std']  = np.std(x[col].values[use])
            y.loc[i,col+'_sum']  = np.sum(x[col].values[use])
            y.loc[i,col+'_cnt']  = use.size
            #Add extra parameters for emerging flux 2018/06/27 J. Prchlik
            if col == 'area_atdiskcenter':
                y.loc[i,col+'_unc'] = np.sqrt(float(np.sum((x[col+'uncert'].values[use])**2))) #Area uncertainty
                y.loc[i,col+'_dtt'] = (x.index[use].max()-x.index[use].min()).total_seconds()/(3600.*24.) #Time in bin in days
    

    #Add time average time offset to days
    #toff = x.index[1:]-x.index[:-1]
    #x.index = x.index+toff/2.
    y.index = y.index+pd.DateOffset(days=14)

    #return y
    return y



#set up time slices
#times when tilts are different
s_d1 = '2012/01/01'
e_d1 = '2012/07/31'
s_d2 = '2013/10/21'
e_d2 = '2015/01/01'

#time when tilts are similiar 
s_s1 = '2011/12/01'
e_s1 = '2013/10/20'


#indices to drop after visual inspection 2018/04/19 J. Prchlik
drop_ind = [10209,10174,11773,9657,9654,7362,12592,13736]

#sampling frequency 
sam = '4W'
#get pandas timeseries representation for filament tracking code time range
#Updated to shorter range for plotting 2018/03/30 J. Prchlik
rng = pd.date_range('2011-06-01 00:00:00','2015-01-01 00:00:00',freq=sam)#.to_timestamp()
#rng = pd.date_range('2012-01-01 00:00:00','2015-01-01 00:00:00',freq=sam)#.to_timestamp()

#read in filament categories given in Brianna's code
#afil = pd.read_pickle('filament_catagories.pic')
#update with hgs coordinates for averages 2018/02/05 J. Prchlik
fil = pd.read_pickle('filament_categories_hgs_mean_l.pic')


#Created length weighted tilt per instance
fil['lw_tilt'] =  fil.fi_length*fil.fi_tilt

#2018/05/10 J. Prchlik
#get number of track instances per unique time
group = fil[['num_inst','event_starttime','track_id','fi_tilt','fi_length','lw_tilt']].groupby(['track_id','event_starttime'])
#count number of unique times per track
track_int = pd.DataFrame(group.size().groupby(level=0).size(),columns=['unq_num'])

#Get the weighted track instance per time tilt (i.e. 1 tilt and length per time period)
wmt_track = pd.DataFrame((group.lw_tilt.sum()/group.fi_length.sum()).groupby(level=0).median(),
                          columns=['wm_med_tilt'])

#Had to hack this way not really sure why 2018/06/14 J. Prchlik
b = (group.fi_length.sum()).groupby(level=0).median()
wmt_track_b = pd.DataFrame(b)
wmt_track_b.rename(columns={'fi_length':'sum_med_length'},inplace=True)

#combine new values into 1 dataframe
wmt_track = wmt_track.join(wmt_track_b)
track_int = track_int.join(wmt_track)

#add back into filament data set
fil = fil.join(track_int)
#remove track id with less than 5 instances
fil = fil.loc[fil.unq_num >= 5,:]

#use the new fi length weight mean tilt median
fil.rename(columns={'med_tilt':'um_med_tilt'},inplace=True)
fil.rename(columns={'wm_med_tilt':'med_tilt'},inplace=True)

#drop specificed indices from inspection 2018/04/19 J. Prchlik
fil.drop(drop_ind,inplace=True)

#Fix that some filaments are missing their track id categories 2018/04/11 J. Prchlik (Need to check if this affects median value still)
# It will not affect media value because we use track id for median value not track id+catagory 2018/04/19 J. Prchlik 
fil.loc[fil[fil.cat_id == 1].track_id.index.unique(),'cat_id'] = 1
fil.loc[fil[fil.cat_id == 2].track_id.index.unique(),'cat_id'] = 2
fil.loc[fil[fil.cat_id == 3].track_id.index.unique(),'cat_id'] = 3
fil.loc[fil[fil.cat_id == 4].track_id.index.unique(),'cat_id'] = 4

#test dynamic time warping
time_warp = False
if time_warp: import mlpy

fil_dict = {}
fil_fmt = 'fil{0:1d}'

fil_keys = ['fil1','fil2','fil3','fil4']

fil_dict['fil1'] = [fil[fil.cat_id == 1],'red'  ,'o','-' ,"Cat. 1"]
fil_dict['fil2'] = [fil[fil.cat_id == 2],'black','x','--',"Cat. 2"]
fil_dict['fil12'] = [fil[((fil.cat_id == 1) | (fil.cat_id == 2))],'red'  ,'o','-' ,"Cat. 1 and 2"]
fil_dict['fil123'] = [fil[((fil.cat_id == 1) | (fil.cat_id == 2) | (fil.cat_id == 3))],'purple','^','-' ,"Cavity"]
fil_dict['fil3'] = [fil[fil.cat_id == 3],'teal' ,'s','-.',"Cat. 3"]
fil_dict['fil4'] = [fil[fil.cat_id == 4],'blue' ,'D','--' ,"No Cavity"]
fil_dict['allf'] = [fil[((fil.cat_id > 0.5) & (fil.cat_id < 4.5))],'blue' ,'D',':' ,"All Filaments"]


#count number of instances in track
#fil_dict[i][0].groupby('track_id').track_id.count() 

#check that the tilt is not decreasing as a function of x value
fig_ti,ax_ti = plt.subplots(figsize=(8,8))

l = ['fil4']

#Get the slope of the tilt and find estimated change in tilt over time
for i in l:
    tilt_slope = fil_dict[i][0].groupby('track_id').apply(lambda v: stats.linregress(v.meanx,v.fi_tilt)[0])
    tilt_range = fil_dict[i][0].groupby('track_id').apply(lambda v: v.meanx.max()-v.meanx.min())
    tilt_maskv = fil_dict[i][0].loc[tilt_slope.index].index.duplicated(keep='first')
    tilt_angle = fil_dict[i][0][~tilt_maskv].med_tilt
    #Plot slopes and average slope offset 
    ax_ti.scatter(tilt_angle.abs(),tilt_slope*tilt_range,marker=fil_dict[i][2],color=fil_dict[i][1],label=fil_dict[i][3])
    ax_ti.errorbar(0.,(tilt_slope*tilt_range).mean(),
                   yerr=3.*(tilt_slope*tilt_range).std()/np.sqrt(tilt_slope.count()),marker='s',color='black')


fancy_plot(ax_ti)

ax_ti.set_ylabel('$\delta$Tilt from Rotation [Deg.]')
ax_ti.set_xlabel('$|$Tilt$|$ [Deg.]')

fig_ti.savefig('plots/tilt_v_x.png',bbox_pad=.1,bbox_inches='tight')
fig_ti.savefig('plots/tilt_v_x.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig_ti)




#setup figures
#fig, ax = plt.subplots(nrows=2,figsize=(4.25,12.5))
#fig, ax = plt.subplots(nrows=2,figsize=(4.25,12.5))
fig, ax = plt.subplots(figsize=(7.75,5.95),ncols=2,sharey=True) #Switched to two variables only 2018/06/14 J. Prchlik Switch to columns for presentation 2018/05/02
fig1, ax1 = plt.subplots(figsize=(11.,8.5))
fig2, ax2 = plt.subplots(figsize=(7.75,5.95),ncols=2,sharey=True) #Switched to two variables only 2018/03/30 J. Prchlik Switch to columns for presentation 2018/05/02
fig4, ax4 = plt.subplots(figsize=(12.5,8.5),ncols=2,sharey=True) #Switched to two variables only 2018/06/14 J. Prchlik Switch to columns for presentation 2018/05/02
#fig5, ax5 = plt.subplots(nrows=2,figsize=(4.25,12))
fig5, ax5 = plt.subplots(figsize=(12.5,8.5),ncols=2,sharey=True) #Switched to two variables only 2018/06/14 J. Prchlik Switch to columns for presentation 2018/05/02
ax2 = ax2.ravel()
#fig.subplots_adjust(hspace=0.001,wspace=0.001)
fig4.subplots_adjust(hspace=0.001)
fig2.subplots_adjust(hspace=0.001,wspace=0.001) #squished together 2018/05/02 J. Prchlik
fig.subplots_adjust(hspace=0.001,wspace=0.001) #squished together 2018/05/02 J. Prchlik
#fig5.subplots_adjust(hspace=0.001,wspace=0.001)

#set up random distribution for comparison
samples = 10000
cuml_tilt = np.linspace(0.,1.,samples)
rand_tilt = np.sort((np.random.rand(samples)-.5)*90.*2.)
norm_tilt = np.random.normal(scale=fil_dict['allf'][0].med_tilt.std(),size=samples)
norm_tilt = np.sort(norm_tilt)



#compare stable vs unstable filaments
#Add north parameter 
stab_keys = ['fil1','fil2','fil12','fil123','fil3','fil4','allf']
for i in stab_keys: 
    d = fil_dict[i]
    d[0]['north'] = 0
    d[0]['north'][d[0].med_l > 0.] = 1
    #remove duplicates early and from all (used to do later) 2018/05/10 
    fil_dict[i][0] = d[0][~d[0].index.duplicated(keep='first')]
    


# get total occurances for a given track
# Not required 2018/04/19 num_inst is correct
# need to check to recount
allf = fil_dict['allf'][0]
#cnt_track = allf.groupby('track_id').track_id.count()


#remove duplicates
allf = allf[~allf.index.duplicated(keep='first')]

#Add between maxima keyword
allf['bet_max'] = 0
allf.set_index(allf.event_starttime_dt,inplace=True)
allf.loc[s_s1:e_s1,'bet_max'] = 1 

#round off the filament length
allf['fi_len_round'] = allf.sum_med_length.map('{:,.3e}'.format)

#parameter to write to tex file
w_p = ['track_id','track_start','track_end','med_l','north','med_tilt','fi_len_round','cat_id','unq_num','bet_max']
#Write important parameters to output file
allf[w_p].sort_values(by=['track_start']).to_latex('ha_filament_table.tex',index_names=False,float_format='%.1f')


#Keys to plot cumlative distribution for 2018/30/30 J. prchlik
cuml_keys = ['fil123','fil4']

#Keys to plot cumlative distribution for 2018/30/30 J. prchlik
for j,i in enumerate(cuml_keys):
    d = fil_dict[i]

    d[0].set_index(d[0]['track_id'],inplace=True)
    d[0] = d[0][~d[0].index.duplicated(keep='last')]

    d[0].set_index(d[0]['event_starttime_dt'],inplace=True)
    d[0].sort_index(inplace=True)

    n = d[0][d[0].north == 1]
    s = d[0][d[0].north == 0]

    n = setup_dis(n)
    s = setup_dis(s)
  
    #overplot statiscal distributions 
    #Remove Random and Gaussian overplots
    #ax2[j].plot(rand_tilt,cuml_tilt,'-',color='teal',label='Random')
    #ax2[j].plot(norm_tilt,cuml_tilt,'-',color='blue',label='Gaussian')

    #Not sure why I did this but it messes up the distributions
    #n.set_index(n['event_starttime_dt'],inplace=True)
    #n.sort_index(inplace=True)
    #s.set_index(s['event_starttime_dt'],inplace=True)
    #s.sort_index(inplace=True)


    #two sample anderson-darling test between n and s of same catagory 
    ad = stats.anderson_ksamp([n.med_tilt.values,s.med_tilt.values])
    k2 = stats.ks_2samp(s.med_tilt.values,n.med_tilt.values)
    
   
    ax[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4],linewidth=5)
    ax[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4],linewidth=5)


    ax4[0].scatter(n.med_tilt,n.med_l,color=d[1],marker=d[2],label=d[4])
    ax4[1].scatter(s.med_tilt,s.med_l,color=d[1],marker=d[2],label=d[4])

    ax1.scatter(d[0].med_l,d[0].med_tilt,color=d[1],marker=d[2],label=d[4])

    ax2[j].plot(n.med_tilt,n.dis,color='red',label='Nothern',linewidth=5)
    ax2[j].plot(s.med_tilt,s.dis,color='black',linestyle='--',label='Southern',linewidth=5)
    #Added printing of mean values to output
    print("##################################################################")
    print(i)
    print('North = ',n.med_tilt.mean(),n.med_tilt.std()/np.sqrt(len(n)))
    print('South = ',s.med_tilt.mean(),s.med_tilt.std()/np.sqrt(len(s)))
    print("##################################################################")
    #removed A-D stat 2018/03/31 J. Prchlik
    #if ad[-1] < 1.0: ax2[j].text(20,.1,'p(A-D) = {0:5.4f}'.format(ad[-1]),fontsize=18)
    #ax2[j].text(12,.15,'p(KS2) = {0:4.3f}'.format(k2[-1]),fontsize=12)
    ax2[j].set_title(d[4])
    ax2[j].set_xlabel('Med. Tilt [Deg.]')
    ax2[j].set_xlim([-95,95])
    fancy_plot(ax2[j])


    #do the camparision for stable filaments vs no stable
    if i == 'fil1':
        #setup for combined 1 and 2 categories 
        d = fil_dict['fil12']
        d[0] = d[0][~d[0].index.duplicated(keep='first')]

        n = d[0][d[0].north == 1]
        s = d[0][d[0].north == 0]
        n = setup_dis(n)
        s = setup_dis(s)

        #setup for combined 1, 2, and 3 categories 
        e = fil_dict['fil123']
        e[0] = e[0][~e[0].index.duplicated(keep='first')]

        n123 = e[0][e[0].north == 1]
        s123 = e[0][e[0].north == 0]
        n123 = setup_dis(n123)
        s123 = setup_dis(s123)


        #plot Med tilt distrbutions 
        #Only plot 1,2,3 combined filaments
        #ax5[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4])
        #ax5[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[0].plot(n123.med_tilt,n123.dis,color=e[1],linestyle=e[3],label=e[4])
        ax5[1].plot(s123.med_tilt,s123.dis,color=e[1],linestyle=e[3],label=e[4])


        #setup d3 and d4 distributions for comparision
        d3 = fil_dict['fil3'][0]
        d3.set_index(d3['track_id'],inplace=True)
        d3.sort_index(inplace=True)
        d3 = d3[~d3.index.duplicated(keep='first')]

        d4 = fil_dict['fil4'][0]
        d4.set_index(d4['track_id'],inplace=True)
        d4.sort_index(inplace=True)
        d4 = d4[~d4.index.duplicated(keep='first')]

        #break d3 and d4 into frames of north and south
        d3n = setup_dis(d3[d3['north'] == 1])
        d3s = setup_dis(d3[d3['north'] == 0])
        d4n = setup_dis(d4[d4['north'] == 1])
        d4s = setup_dis(d4[d4['north'] == 0])

        #two sample anderson-darling test between n or s of differnt catagories
        ad3n = stats.anderson_ksamp([d3n.med_tilt.values,n.med_tilt.values])
        k23n = stats.ks_2samp(d3n.med_tilt.values,n.med_tilt.values)
        ad3s = stats.anderson_ksamp([d3s.med_tilt.values,s.med_tilt.values])
        k23s = stats.ks_2samp(d3s.med_tilt.values,s.med_tilt.values)

        #two sample anderson-darling test between n or s of 1, 2, and 3 vs 4
        ad4n = stats.anderson_ksamp([d4n.med_tilt.values,n123.med_tilt.values])
        k24n = stats.ks_2samp(d4n.med_tilt.values,n123.med_tilt.values)
        ad4s = stats.anderson_ksamp([d4s.med_tilt.values,s123.med_tilt.values])
        k24s = stats.ks_2samp(d4s.med_tilt.values,s123.med_tilt.values)

        #show fit stat on plot for 1 and 2 vs. 3
        #remove A-D stat 2018/03/30 J. Prchlik
        #if ad[-1] < 1.0: ax5[0].text(5,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3n[-1]),fontsize=14)
        ax5[0].text(5,.15,'p(KS2;12,3) = {0:4.3f}'.format(k23n[-1]),fontsize=14)
        #if ad[-1] < 1.0: ax5[1].text(5,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3s[-1]),fontsize=14)
        ax5[1].text(5,.15,'p(KS2;12,3) = {0:4.3f}'.format(k23s[-1]),fontsize=14)

        #show fit stat on plot for 1, 2, and 3 vs 4
        #if ad[-1] < 1.0: ax5[0].text(5,.01,'p(A-D;123,4) = {0:5.4f}'.format(ad4n[-1]),fontsize=14)
        ax5[0].text(5,.05,'p(KS2;123,4) = {0:5.4f}'.format(k24n[-1]),fontsize=14)
        #if ad[-1] < 1.0: ax5[1].text(5,.01,'p(A-D;123,4) = {0:5.4f}'.format(ad4s[-1]),fontsize=14)
        ax5[1].text(5,.05,'p(KS2;123,4) = {0:5.4f}'.format(k24s[-1]),fontsize=14)


    elif ((i == 'fil3') | (i == 'fil4')):
        ax5[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4])




#get person r value for ax4[0] (north) and ax4[1] (south)
allf = fil_dict['allf'][0]
allf.set_index(allf['event_starttime_dt'],inplace=True)
allf.sort_index(inplace=True)
   

npp = stats.pearsonr(allf[allf.north == 1].med_tilt.values,allf[allf.north == 1].med_l.values)
spp = stats.pearsonr(allf[allf.north == 0].med_tilt.values,allf[allf.north == 0].med_l.values)
ax4[0].text(-90,20,'r={0:4.3f},p={0:4.3f}'.format(*npp),fontsize=12)
ax4[1].text(-90,-20,'r={0:4.3f},p={0:4.3f}'.format(*spp),fontsize=12)

#Cut down dow filaments 1,2,3 versus 4 2018/03/30 J. Prchlik
#array of filament objects
#Cut down dow filaments 1,2,3 versus 4 2018/03/30 J. Prchlik
tilt_time = ['fil123','fil4']
plot_rows = len(tilt_time)+1
#There for only need 3 rows instead of 4
#plotting 1and 2, 3, and 4 versus time and sunspots
#update fig size for 3 plots
fig_size = (11.,11.3)
fig3, ax3 = plt.subplots(figsize=fig_size,nrows=plot_rows,sharex=True)
fig3.subplots_adjust(hspace=0.001,wspace=0.001)
#plotting 1and 2, 3, and 4 versus time and emerging flux
fig8, ax8 = plt.subplots(figsize=fig_size,nrows=plot_rows,sharex=True)
fig8.subplots_adjust(hspace=0.001,wspace=0.001)

#plots for 1 and 2, 3, 5 versus time and ar PIL curvature
fig10, ax10 = plt.subplots(figsize=fig_size,nrows=plot_rows,sharex=True)
fig10.subplots_adjust(hspace=0.001,wspace=0.001)

#plots for 1 and 2, 3, 5 versus time and Sigmoid properties curvature
fig11, ax11 = plt.subplots(figsize=fig_size,nrows=plot_rows,sharex=True)
fig11.subplots_adjust(hspace=0.001,wspace=0.001)

#plots for 1 and 2, 3, 5 versus time and ar height
fig12, ax12 = plt.subplots(figsize=fig_size,nrows=plot_rows,sharex=True)
fig12.subplots_adjust(hspace=0.001,wspace=0.001)


for j,i in enumerate(tilt_time):

    allf = fil_dict[i][0]
    allf.set_index(allf['track_id'],inplace=True)
    #get unique indices 
    allf = allf[~allf.index.duplicated(keep='first')]
    allf.set_index(allf['event_starttime_dt'],inplace=True)
    allf.sort_index(inplace=True)

    #split into noth and south
    #http://benalexkeen.com/resampling-time-series-data-with-pandas/
    #get running mean
    bn = allf[allf.north == 1]
    bs = allf[allf.north == 0]

    #get running mean
    ###mbn = bn.resample(sam).mean()
    ###mbs = bs.resample(sam).mean()
    ####get running standard deviation
    ###sbn = bn.resample(sam).std()
    ###sbs = bs.resample(sam).std()
    ####get running count
    ###cbn = bn.resample(sam).count()
    ###cbs = bs.resample(sam).count()
  
    #resample with fixed cadence
    mbn = real_resamp(bn,rng)
    mbs = real_resamp(bs,rng)
    
    #get error including counting errors for north and south pole
    tot_err_n = np.sqrt((mbn.med_tilt_std.values/np.sqrt(mbn.med_tilt_cnt.values))**2.)#+(mbn.med_tilt_mean.values/np.sqrt(mbn.med_tilt_cnt.size))**2.)
    tot_err_s = np.sqrt((mbs.med_tilt_std.values/np.sqrt(mbs.med_tilt_cnt.values))**2.)#+(mbs.med_tilt_mean.values/np.sqrt(mbs.med_tilt_cnt.size))**2.)


    #Do dynamic time warping and plot the result
    if ((i == 'fil4') & (time_warp)): 

        fig100,ax100 = plt.subplots()

        #array to convert to dates for plotting time series from dtw
        daterange = np.array([mbs.dropna().index.min(),mbs.dropna().index.max(),mbn.dropna().index.min(),mbn.dropna().index.max()])
        date_num = mdates.date2num(daterange)


        #Southern times
        s_time = mdates.date2num(mbs.med_tilt_mean.dropna().index.to_pydatetime())
        n_time = mdates.date2num(mbn.med_tilt_mean.dropna().index.to_pydatetime())

        #get differneces in time
        ds_t = np.diff(s_time)
        dn_t = np.diff(n_time)

        #Add last element twice
        ds_t = np.append(ds_t,ds_t[-1])
        dn_t = np.append(dn_t,dn_t[-1])

        #calculate dtw
        dist, cost, path = mlpy.dtw_std(mbn.med_tilt_mean.dropna().abs().values,mbs.med_tilt_mean.dropna().abs().values,dist_only=False)

        #variabiles to plot dtw path 
        n_time = mdates.date2num(mbn.med_tilt_mean.dropna().index[path[0]].to_pydatetime())
        s_time = mdates.date2num(mbs.med_tilt_mean.dropna().index[path[1]].to_pydatetime())

        #tell matplotlib x and y axis are dates
        ax100.xaxis_date()
        ax100.yaxis_date()

        #plot dtw
        #Removed Cost matrix 2018/01/19 J. Prchlik
        #plot1 = ax100.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest',
        #                     extent=date_num,aspect='auto')
        #plot2 = ax100.plot(ds_t[path[1]]+s_time[path[1]], dn_t[path[0]]+n_time[path[0]], 'w')
        #plot4 = ax100.plot(ds_t[path[1]]+s_time[path[1]], dn_t[path[0]]+n_time[path[0]], 'b',linewidth=3)
        plot4 = ax100.plot(s_time,n_time, 'b',linewidth=3)
        plot3 = ax100.plot([date_num[0],date_num[1]],[date_num[2],date_num[3]], 'r')


        #plot axis back it YYYY-MM-DD format
        date_format = mdates.DateFormatter('%Y-%m')

        #change J date describtion on axis
        ax100.xaxis.set_major_formatter(date_format)
        ax100.yaxis.set_major_formatter(date_format)

        #set x and y limits 
        ax100.set_xlim([date_num[0],date_num[1]])
        ax100.set_ylim([date_num[2],date_num[3]])

        #rotate to 45 degrees
        fig100.autofmt_xdate()

        #set up labels
        ax100.set_xlabel('Southern Time')
        ax100.set_ylabel('Northern Time')

        #xlim = ax100.set_xlim((-0.5, cost.shape[0]-0.5))
        #ylim = ax100.set_ylim((-0.5, cost.shape[1]-0.5))
 

    #make similar plots for sunspots and emerging flux and active regions with filaments
    for rax in [ax3[j],ax8[j],ax10[j],ax11[j],ax12[j]]:
    #plot running mean
        rax.errorbar(mbn.index,mbn.med_tilt_mean,xerr=timedelta(days=14),yerr=tot_err_n,capsize=3,barsabove=True,fmt='-',color='red',linewidth=3,label='Northern Mean ({0})'.format(sam))
        rax.errorbar(mbs.index,mbs.med_tilt_mean,xerr=timedelta(days=14),yerr=tot_err_s,capsize=3,barsabove=True,fmt='--',color='black',linewidth=3,label='Southern Mean ({0})'.format(sam))
        
        #Make tilt versus time plot
        #remove so plot is more clear for presentation 2018/05/12 J. Prchlik
        ###rax.scatter(bn.index,bn.med_tilt,color='magenta',marker='o',label='Northern')
        ###rax.scatter(bs.index,bs.med_tilt,color='grey',marker='D',label='Southern')

        #Add different and similar lines 2018/03/30 J. Prchlik
        #Remove dashed lines for solar cycle 2018/05/10 J. Prchlik
        ###rax.axvline(mdates.date2num(pd.to_datetime(s_s1)),linestyle='-.',color='blue',alpha=0.6)
        rax.axvline(mdates.date2num(pd.to_datetime(e_s1)),linestyle='-.',color='blue',alpha=0.6)


        #Y title
        #Update width 1, 2, and 3 (i.e. Have a cavity compbined)
        rax.set_ylabel("Med. Tilt [Deg.]\n {0}".format(fil_dict[i][4]))
        fancy_plot(rax)
        rax.set_ylim([-90.,90.])


#Take advatage of the fact I use filament 4 last in the looping and find the correlations between sunspot height and filament 4 tilts
fig6, ax6 = plt.subplots()

#Add sunspot number to output
#### 2018/01/19 Added sunspot number from NOAA to plots
#ss_nm_hist = pd.read_csv('/Volumes/Pegasus/jprchlik/dscovr/solar_wind_events/sun_spot_number/SN_m_tot_V2.0.txt')
#ss_nm_hist = pd.read_csv('/Volumes/Pegasus/jprchlik/dscovr/solar_wind_events/sun_spot_number/SN_m_tot_V2.0.txt')
#from royal Belgium observatory so you know its good 2018/02/02 J. Prchlik
ss_nm = pd.read_csv('sunspots/SN_m_hem_V2.0.csv',names=['year','month','year_frac','t_ss','n_ss','s_ss','t_ss_unc','n_ss_unc','s_ss_unc','t_ss_c','n_ss_t','s_ss_t'],sep=';',index_col=False)
ss_nm['time_dt'] =  pd.to_datetime(ss_nm.year.astype(str)+'/'+ss_nm.month.astype(str)+'/15')
ss_nm.set_index(ss_nm.time_dt,inplace=True)

#ss_nm = pd.read_pickle('sunspots/query_output/all_ss_20120101-20141130.pic')
##cut to eruptions only above 30 degrees latitude 
#n_ss = ss_nm[ss_nm.hgs_y >  0.]
#s_ss = ss_nm[ss_nm.hgs_y < -0.]
#

##bin up in 4W bins 
#bn_ss = real_resamp(n_ss,rng,col='hgs_y')
#bs_ss = real_resamp(s_ss,rng,col='hgs_y')
#
##errors on sunspot number including counting
#tot_err_s = np.sqrt((bs_ss.hgs_y_std.values/np.sqrt(bs_ss.hgs_y_cnt.values))**2.)#+(bs_ss.hgs_y_mean.values/np.sqrt(bs_ss.hgs_y_cnt.size))**2) #errors due to counting
#tot_err_n = np.sqrt((bn_ss.hgs_y_std.values/np.sqrt(bn_ss.hgs_y_cnt.values))**2.)#+(bn_ss.hgs_y_mean.values/np.sqrt(bn_ss.hgs_y_cnt.size))**2) #errors due to counting
tot_err_s = ss_nm.s_ss_unc
tot_err_n = ss_nm.n_ss_unc
                     
                     
#plot average height of sunspots
#updated with new plot_rows parameter 2018/03/30 J. Prchlik
ax3[plot_rows-1].errorbar(ss_nm.index,ss_nm.n_ss.values,yerr=tot_err_n,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='s',color='red',label='Northern ({0})'.format(sam))
ax3[plot_rows-1].errorbar(ss_nm.index,ss_nm.s_ss.values,yerr=tot_err_s,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax3[plot_rows-1].plot(ss_nm.index,ss_nm.n_ss.values,'-',color='red',label='Northern ({0})'.format(sam))
ax3[plot_rows-1].plot(ss_nm.index,ss_nm.s_ss.values,'--',color='black',label='Southern ({0})'.format(sam))

#Add different and similar lines 2018/03/30 J. Prchlik
#Remove dashed lines for solar cycle 2018/05/10 J. Prchlik
###ax3[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(s_s1)),linestyle='-.',color='blue',alpha=0.6)
ax3[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(e_s1)),linestyle='-.',color='blue',alpha=0.6)

#Add tick label rotations for datetime
for tick in ax3[plot_rows-1].get_xticklabels():
    tick.set_rotation(25)


#Northern Matching
#ax6.scatter(np.abs(bn_ss.hgs_y_mean.values),np.abs(mbn.med_tilt_mean),color='red',marker='o',label='North')
#ax6.scatter(np.abs(bn_ss.hgs_y_mean.values)[1:],np.abs(mbn.med_tilt_mean.values[:-1]),color='red',marker='<',label='FI -14 days')
#Commented out J. prchlik 2018/02/02
#ax6.scatter(np.abs(bn_ss.hgs_y_mean.values)[:-1],-(mbn.med_tilt_mean.values[1:]),color='red',marker='o',label='Northern (FI +28 days)')

#Southern Matching
#ax6.scatter(np.abs(bs_ss.hgs_y_mean.values),np.abs(mbs.med_tilt_mean),color='black',marker='o',label='Southern')
#ax6.scatter(np.abs(bs_ss.hgs_y_mean.values)[1:],np.abs(mbs.med_tilt_mean.values[:-1]),color='black',marker='<',label='FI -14 days')
#Commented out J. prchlik 2018/02/02
#ax6.scatter(np.abs(bs_ss.hgs_y_mean.values)[:-1],(mbs.med_tilt_mean.values[1:]),color='black',marker='o',label='Southern (FI +28 days)')

#Commented block out J. prchlik 2018/02/02
#x = np.concatenate([np.abs(bn_ss.hgs_y_mean.values)[:-1],np.abs(bs_ss.hgs_y_mean.values)[:-1]])
#y = np.concatenate([-mbn.med_tilt_mean.values[1:],mbs.med_tilt_mean.values[1:]])
#
#use, = np.where((np.isfinite(x)) & (np.isfinite(y)))
#
#r_tl_ss = stats.pearsonr(x[use],y[use])
#
#ax6.text(5.,-50.,'r={0:4.3f}'.format(*r_tl_ss),fontsize=16,color='black')
#
#ax6.set_xlim([4.,25.])
#
#ax6.set_xlabel('$|$Mean SS Lat.$|$ [deg.]')
#ax6.set_ylabel('Mean FI Tilt [deg.]')


##
# plot the difference between sunspot height and filament tilt in the N and S
#Commented block out J. prchlik 2018/02/02
###fig7, ax7 = plt.subplots()
###
####get error including number using Poisson stats
###tot_err_x = np.sqrt((bs_ss.hgs_y_std.values/np.sqrt(bs_ss.hgs_y_cnt.values))**2.+(bn_ss.hgs_y_std.values/np.sqrt(bn_ss.hgs_y_cnt.values))**2.)#+
###                     #(bs_ss.hgs_y_mean.values/np.sqrt(bs_ss.hgs_y_cnt.size))**2+ #errors due to counting
###                     #(bn_ss.hgs_y_mean.values/np.sqrt(bn_ss.hgs_y_cnt.size))**2) #errors due to counting
###tot_err_y = np.sqrt((mbs.med_tilt_std.values/np.sqrt(mbs.med_tilt_cnt.values))**2.+(mbn.med_tilt_std.values/np.sqrt(mbn.med_tilt_cnt.values))**2.)#+
###                     #(mbs.med_tilt_mean.values/np.sqrt(mbs.med_tilt_cnt.size))**2+ #errors due to counting
###                     #(mbn.med_tilt_mean.values/np.sqrt(mbn.med_tilt_cnt.size))**2) #errors due to counting
###
###ax7.scatter(bn_ss.hgs_y_mean.values+bs_ss.hgs_y_mean.values,mbn.med_tilt_mean.values-mbs.med_tilt_mean.values,color='black',marker='o')
###ax7.errorbar(bn_ss.hgs_y_mean.values+bs_ss.hgs_y_mean.values,mbn.med_tilt_mean.values-mbs.med_tilt_mean.values,color='black',
###             yerr=tot_err_y,xerr=tot_err_x,capsize=3,barsabove=True,linewidth=3,fmt='o')
###
###ax7.set_ylabel('Diff. Tilt (N-S) [deg.]')
###ax7.set_xlabel('Diff. SS Lat. (N-S) [deg.]')


###############
#plots for emerging flux
#Add sunspot number to output
ef_nm = pd.read_pickle('emerging_flux/query_output/all_ef_20120101-20141130.pic')
#Add earlier observations
ef_2  = pd.read_pickle('emerging_flux/query_output/all_ef_20100523-20120101.pic')
#Add a month of later observations
ef_3  = pd.read_pickle('emerging_flux/query_output/all_ef_20141130-20150101.pic')
#add earlier emerging flux observations to ones during the filament catalog
ef_nm = pd.concat([ef_2,ef_nm,ef_3])

#Double check events are unique
ef_nm.drop_duplicates(subset=['SOL_standard'], keep='first', inplace=True)


#Just get HMI emergin flux 
ef_nm = ef_nm[((ef_nm.obs_channelid == 'LOS Magnetograms') & (ef_nm.frm_humanflag == 'false') & (ef_nm.search_instrument == 'HMI') & (ef_nm.search_frm_name == 'SWAMIS-EF')) ]

#remove largest ef area because it is a false positive 2018/07/16 J. Prchlik
bad_index = ef_nm.query('SOL_standard == "SOL2014-11-21T05:47:15L261C100"').index
ef_nm.drop(bad_index,inplace=True)

#Get flux in emerging flux areas
ef_nm.loc[:,'sum_unsigned_flux'] = ef_nm.ef_sumpossignedflux-ef_nm.ef_sumnegsignedflux
ef_nm.loc[:,'sum_signed_flux']   = ef_nm.ef_sumpossignedflux+ef_nm.ef_sumnegsignedflux

#cut to eruptions only above 30 degrees latitude 
n_ef = ef_nm[ef_nm.hgs_y >  0.]
s_ef = ef_nm[ef_nm.hgs_y < -0.]

#check = 'hgs_y'
#check = 'ef_axislength'
#check = 'sum_unsigned_flux'
#ratio almost looks like it has a year offset
check = 'ef_proximityratio'
check = 'ef_axislength'

#get area of EF at DC 2018/06/27
check = 'area_atdiskcenter' # in km^2
#check = 'ef_sumpossignedflux'
#check = 'ef_sumpossignedflux'

#bin up in 4W bins 
bn_ef = real_resamp(n_ef,rng,col=check)
bs_ef = real_resamp(s_ef,rng,col=check)

#errors on sunspot number including counting
#tot_err_s = np.sqrt((bs_ef[check+'_std'].values/np.sqrt(bs_ef[check+'_cnt'].values))**2.)
#tot_err_n = np.sqrt((bn_ef[check+'_std'].values/np.sqrt(bn_ef[check+'_cnt'].values))**2.)
#Added newly calculated errors based on sum of square reported errors in Mm^2
tot_err_s = bn_ef[check+'_unc']/bn_ef[check+'_dtt']*1e-12
tot_err_n = bs_ef[check+'_unc']/bs_ef[check+'_dtt']*1e-12



#plot average height of emerging flux
#update with new plot rows parameter 2018/03/30 J. Prchlik
#Add new units of plot in Mm^2 per day
ax8[plot_rows-1].errorbar(bn_ef.index,1e-12*bn_ef[check+'_sum'].values/bn_ef[check+'_dtt'].values,yerr=tot_err_n,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='s',color='red',label='Northern ({0})'.format(sam))
ax8[plot_rows-1].errorbar(bs_ef.index,1e-12*bs_ef[check+'_sum'].values/bs_ef[check+'_dtt'].values,yerr=tot_err_s,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax8[plot_rows-1].plot(bn_ef.index,1.e-12*bn_ef[check+'_sum'].values/bn_ef[check+'_dtt'].values,'-',color='red',label='Northern ({0})'.format(sam))
ax8[plot_rows-1].plot(bs_ef.index,1.e-12*bs_ef[check+'_sum'].values/bs_ef[check+'_dtt'].values,'--',color='black',label='Southern ({0})'.format(sam))

ax8[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(e_s1)),linestyle='-.',color='blue',alpha=0.6)
fancy_plot(ax8[plot_rows-1])

ax8[plot_rows-1].set_ylabel('EF Area \n [Gm$^2$/day]')
ax8[plot_rows-1].set_xlabel('Time [UTC]')

ax8[0].legend(loc='upper left',scatterpoints=1,frameon=False,fontsize=18)
#Add tick label rotations for datetime
for tick in ax8[plot_rows-1].get_xticklabels():
    tick.set_rotation(25)

fig8.savefig('plots/emerging_flux_time.png',bbox_pad=.1,bbox_inches='tight')
fig8.savefig('plots/emerging_flux_time.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig8)

###########################################
###########################################
#plots for Active region properties
#Add sunspot number to output
ar_nm = pd.read_pickle('active_regions/query_output/all_ar_20100522-20141130_real.pic')

#sort index 
ar_nm.sort_index(inplace=True)

#Remove None columns in sunspots
ar_nm = ar_nm[ar_nm.ar_numspots.astype(str) != 'None']

#change ar_numspots to numeric
ar_nm['ar_numspots'] = pd.to_numeric(ar_nm.ar_numspots)

#create a copy to add datetime index back in
ar_fn = ar_nm.copy()
ar_fn.set_index(ar_fn.ar_noaanum,inplace=True)

#Group by AR number and get the median value
ar_nm = ar_nm.groupby('ar_noaanum').mean()

#reset index for ar grouping
ar_nm['event_starttime'] = pd.to_datetime(ar_fn[~ar_fn.index.duplicated(keep='first')].event_starttime)
ar_nm.set_index('event_starttime',inplace=True)

#cut to eruptions only above 30 degrees latitude 
n_ar = ar_nm[ar_nm.hgs_y >  0.]
s_ar = ar_nm[ar_nm.hgs_y < -0.]

#parameter to compare with tilt
#check = 'ar_polarity'
#check = 'meanshearangle'
#check = 'meaninclinationgamma' #same thing as shear angle
#check = 'unsignedflux'
#check = 'meanvertcurrentdensity'
#check = 'ar_axislength'
#check = 'meantwistalpha' #a couple bumps about a year before the filament tilt angles
#check = 'highsheararea'
#check = 'ar_neutrallength'
#check = 'ar_polarity'
check = 'ar_numspots'

#bin up in 4W bins 
bn_ar = real_resamp(n_ar,rng,col=check)
bs_ar = real_resamp(s_ar,rng,col=check)

#errors on sunspot number including counting
tot_err_s = np.sqrt((bs_ar[check+'_std'].values/np.sqrt(bs_ar[check+'_cnt'].values))**2.)
tot_err_n = np.sqrt((bn_ar[check+'_std'].values/np.sqrt(bn_ar[check+'_cnt'].values))**2.)

#plot average height of emerging flux
ax10[plot_rows-1].errorbar(bn_ar.index,np.abs(bn_ar[check+'_sum'].values),yerr=tot_err_n,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='s',color='red',label='Northern ({0})'.format(sam))
ax10[plot_rows-1].errorbar(bs_ar.index,np.abs(bs_ar[check+'_sum'].values),yerr=tot_err_s,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax10[plot_rows-1].plot(bn_ar.index,np.abs(bn_ar[check+'_sum'].values),'-',color='red',label='Northern ({0})'.format(sam))
ax10[plot_rows-1].plot(bs_ar.index,np.abs(bs_ar[check+'_sum'].values),'--',color='black',label='Southern ({0})'.format(sam))

#Add different and similar lines 2018/03/30 J. Prchlik        
#Removed lines signifying solar cycle 2018/05/10 J. Prchlik
###ax10[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(s_s1)),linestyle='-.',color='blue',alpha=0.6)
###ax10[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(e_s1)),linestyle='-.',color='blue',alpha=0.6)

fancy_plot(ax10[plot_rows-1])

ax10[plot_rows-1].set_ylabel('Ave. Sunspots [\#]')
ax10[plot_rows-1].set_xlabel('Time [UTC]')
ax10[0].legend(loc='upper left',scatterpoints=1,frameon=False)

fig10.savefig('plots/ar_sunspots_time.png',bbox_pad=.1,bbox_inches='tight')
fig10.savefig('plots/ar_sunspots_time.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig10) 

#Add Sunspot number in North and South to DTW plots
#2018/01/19 J. Prchlik
#Switch to Belgium Sunspot number 2018/02/05  J. Prchlik

#Cut DTW range down 2018/02/05

#Calculate DTW
#Removed required DTW 2018/03/30
if time_warp:
    dist_ar, cost_ar, path_ar = mlpy.dtw_std(ss_nm['2011/06/01':'2015/06/01'].n_ss.abs().values,ss_nm['2011/06/01':'2015/06/01'].s_ss.abs().values,dist_only=False)
    #Southern times
    n_ar_time = mdates.date2num(ss_nm['2011/06/01':'2015/06/01'].iloc[path_ar[0],:].index.to_pydatetime())
    s_ar_time = mdates.date2num(ss_nm['2011/06/01':'2015/06/01'].iloc[path_ar[1],:].index.to_pydatetime())


    plot7 = ax100.plot(s_ar_time, n_ar_time,'black',linewidth=3)

    #ax100.set_xlim([s_ar_time[0],s_ar_time[-1]])
    #ax100.set_ylim([n_ar_time[0],n_ar_time[-1]])
    #manually show plotted range
    time_range = [datetime(2011,6,1),datetime(2015,6,1)]
    ax100.set_xlim(time_range)
    ax100.set_ylim(time_range)
    
    
    ax100.grid(True,color='gray',linestyle='--')
    
    fig100.savefig('plots/time_warp_fil4_tilt.png',bbox_pad=.1,bbox_inches='tight')
    fig100.savefig('plots/time_warp_fil4_tilt.eps',bbox_pad=.1,bbox_inches='tight')
    plt.close(fig100)


#Also check sunspot height using AR indentifier
check = 'hgs_y'

#bin up in 4W bins 
bn_ar = real_resamp(n_ar,rng,col=check)
bs_ar = real_resamp(s_ar,rng,col=check)

#errors on sunspot number including counting
tot_err_s = np.sqrt((bs_ar[check+'_std'].values/np.sqrt(bs_ar[check+'_cnt'].values))**2.)
tot_err_n = np.sqrt((bn_ar[check+'_std'].values/np.sqrt(bn_ar[check+'_cnt'].values))**2.)

#plot average height of emerging flux
ax12[plot_rows-1].errorbar(bn_ar.index,np.abs(bn_ar[check+'_mean'].values),yerr=tot_err_n,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='s',color='red',label='Northern ({0})'.format(sam))
ax12[plot_rows-1].errorbar(bs_ar.index,np.abs(bs_ar[check+'_mean'].values),yerr=tot_err_s,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax12[plot_rows-1].plot(bn_ar.index,np.abs(bn_ar[check+'_mean'].values),'-',color='red',label='Northern ({0})'.format(sam))
ax12[plot_rows-1].plot(bs_ar.index,np.abs(bs_ar[check+'_mean'].values),'--',color='black',label='Southern ({0})'.format(sam))
ax12[plot_rows-1].scatter(n_ar.index,np.abs(n_ar[check].values),marker='o',color='red',label=None)
ax12[plot_rows-1].scatter(s_ar.index,np.abs(s_ar[check].values),marker='D',color='black',label=None)

#Add different and similar lines 2018/03/30 J. Prchlik        
#Removed lines signfying different parts of the solar cycle
###ax12[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(s_s1)),linestyle='-.',color='blue',alpha=0.6)
###ax12[plot_rows-1].axvline(mdates.date2num(pd.to_datetime(e_s1)),linestyle='-.',color='blue',alpha=0.6)

fancy_plot(ax12[plot_rows-1])

ax12[plot_rows-1].set_ylabel('Ave. AR Lat. [Deg.]')
ax12[plot_rows-1].set_xlabel('Time [UTC]')

fig12.savefig('plots/ar_height_time.png',bbox_pad=.1,bbox_inches='tight')
fig12.savefig('plots/ar_height_time.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig12)

###########################################
###########################################
#

###########################################
###########################################
#
sg_nm = pd.read_pickle('sigmoids/query_output/all_sg_20101208-20141130.pic')
#sort index 
sg_nm.sort_index(inplace=True)

#use only automatic sigmoids
sg_nm = sg_nm[sg_nm.frm_humanflag == 'false']

#HCR check (not sure of meaining but why not for now)
sg_nm = sg_nm[sg_nm.hcr_checked == 'true']

#Use 131 for observation for now
sg_nm = sg_nm[sg_nm.obs_channelid == '131_THIN']

#already cut to only eds observer but drive the point home
sg_nm = sg_nm[sg_nm.kb_archivist == 'eds']

#Cause a ~40% reduction in sample but I cant think of a fast way to combine sigmoid observations 
#without using the active region number
sg_nm = sg_nm[sg_nm.ar_noaanum > 1.]

#manually calculate sigmoid tilt using theil slopes
sg_nm['hpc_bbox_p'] = [loads(i) for i in sg_nm.hpc_bbox.values]
sg_nm['sg_slope'] = [stats.theilslopes(i.exterior.coords.xy)[0] for i in sg_nm.hpc_bbox_p.values]
sg_nm['sg_height'] = [np.ptp(i.exterior.coords.xy[1]) for i in sg_nm.hpc_bbox_p.values]

#use the bounding box of the sigmoid to get a rough tilt 
sg_nm['sg_tilt'] = -np.sin(sg_nm.sg_slope.values/sg_nm.sg_height.values)*180./np.pi

#create a copy to add datetime index back in
sg_fn = sg_nm.copy()
sg_fn.set_index(sg_fn.ar_noaanum,inplace=True)


#Group by AR number and get the median value
sg_nm = sg_nm.groupby('ar_noaanum').median()

#add datetime back in and set index
sg_nm['event_starttime'] = pd.to_datetime(sg_fn[~sg_fn.index.duplicated(keep='first')].event_starttime)
sg_nm.set_index('event_starttime',inplace=True)



check = 'sg_tilt' 

#cut to eruptions only above 30 degrees latitude 
n_sg = sg_nm[sg_nm.hgs_y >  0.]
s_sg = sg_nm[sg_nm.hgs_y < -0.]

#bin up in 4W bins 
bn_sg = real_resamp(n_sg,rng,col=check)
bs_sg = real_resamp(s_sg,rng,col=check)

#add errors
tot_err_s = np.sqrt((bs_sg[check+'_std'].values/np.sqrt(bs_sg[check+'_cnt'].values))**2.)
tot_err_n = np.sqrt((bn_sg[check+'_std'].values/np.sqrt(bn_sg[check+'_cnt'].values))**2.)

#plot average height of emerging flux
ax11[plot_rows-1].errorbar(bn_sg.index,bn_sg[check+'_mean'].values,yerr=tot_err_n,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='s',color='red',label='Northern ({0})'.format(sam))
ax11[plot_rows-1].errorbar(bs_sg.index,bs_sg[check+'_mean'].values,yerr=tot_err_s,xerr=timedelta(days=14),capsize=3,barsabove=True,linewidth=3,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax11[plot_rows-1].plot(bn_sg.index,bn_sg[check+'_mean'].values,'-',color='red',label='Northern ({0})'.format(sam))
ax11[plot_rows-1].plot(bs_sg.index,bs_sg[check+'_mean'].values,'--',color='black',label='Southern ({0})'.format(sam))
ax11[plot_rows-1].scatter(n_sg.index,n_sg[check].values,color='red',label=None)
ax11[plot_rows-1].scatter(s_sg.index,s_sg[check].values,color='black',label=None)

fancy_plot(ax11[plot_rows-1])

ax11[plot_rows-1].set_ylabel('Sigmoid Tilt [Deg.]')
ax11[plot_rows-1].set_xlabel('Time [UTC]')

#ax11[3].set_ylim([0.0,2.])
fig11.savefig('plots/sg_curvature_time.png',bbox_pad=.1,bbox_inches='tight')
fig11.savefig('plots/sg_curvature_time.eps',bbox_pad=.1,bbox_inches='tight')
###########################################
###########################################

#Plot Cumulative distributions of filaments in times where there is a difference between north and south
fig9, ax9 = plt.subplots(figsize=(6,6))

cat4 = fil_dict['fil4'][0]

#change index to track id
cat4.set_index(cat4['track_id'],inplace=True)
#get unique track indices 
cat4 = cat4[~cat4.index.duplicated(keep='first')]
#set index to time 
cat4.set_index(cat4['event_starttime_dt'],inplace=True)
cat4.sort_index(inplace=True)



#separate into north and south
n_cat4 = cat4[cat4.north == 1]
s_cat4 = cat4[cat4.north == 0]

#separate into different and similar times
#time when different
n_cat4_d = n_cat4[((n_cat4.index <= e_d1) | (n_cat4.index >= s_d2))]
s_cat4_d = s_cat4[((s_cat4.index <= e_d1) | (s_cat4.index >= s_d2))]

#time when similar  
n_cat4_s = n_cat4[s_s1:e_s1]
s_cat4_s = s_cat4[s_s1:e_s1]

#plot cumlative distributions
#first sort
n_cat4_d.sort_values('med_tilt',inplace=True)
n_cat4_s.sort_values('med_tilt',inplace=True)
s_cat4_d.sort_values('med_tilt',inplace=True)
s_cat4_s.sort_values('med_tilt',inplace=True)

#then create cumulative fraction
n_cat4_d['frac'] = np.arange(0,len(n_cat4_d))/float(len(n_cat4_d))
n_cat4_s['frac'] = np.arange(0,len(n_cat4_s))/float(len(n_cat4_s))
s_cat4_d['frac'] = np.arange(0,len(s_cat4_d))/float(len(s_cat4_d))
s_cat4_s['frac'] = np.arange(0,len(s_cat4_s))/float(len(s_cat4_s))

#finally plot
ax9.plot(n_cat4_d.med_tilt,n_cat4_d.frac,linestyle='-',color='red',label='North Max.' )
ax9.plot(n_cat4_s.med_tilt,n_cat4_s.frac,linestyle='-.',color='magenta',label='North Norm.' )
ax9.plot(s_cat4_d.med_tilt,s_cat4_d.frac,linestyle='-',color='black',label='South Max.' )
ax9.plot(s_cat4_s.med_tilt,s_cat4_s.frac,linestyle='-.',color='gray',label='South Norm.' )


#Add A-D stats to plot
#calculate and plot A-D stats
ad_ns_s =stats.anderson_ksamp([n_cat4_s.med_tilt.values,s_cat4_s.med_tilt.values],midrank=False)
ad_ns_d =stats.anderson_ksamp([n_cat4_d.med_tilt.values,s_cat4_d.med_tilt.values],midrank=False)

ad_nn_d =stats.anderson_ksamp([n_cat4_d.med_tilt.values,n_cat4_d.med_tilt.values],midrank=False)
ad_ss_d =stats.anderson_ksamp([s_cat4_d.med_tilt.values,s_cat4_d.med_tilt.values],midrank=False)

#ad_nn_s =stats.anderson_ksamp([n_cat4_s.med_tilt.values,n_cat4_s.med_tilt.values])
#ad_ss_s =stats.anderson_ksamp([s_cat4_s.med_tilt.values,s_cat4_s.med_tilt.values])

#ks_two statistic
k2_ns_s =stats.ks_2samp(n_cat4_s.med_tilt.values,s_cat4_s.med_tilt.values)
k2_ns_d =stats.ks_2samp(n_cat4_d.med_tilt.values,s_cat4_d.med_tilt.values)

k2_nn_d =stats.ks_2samp(n_cat4_s.med_tilt.values,n_cat4_d.med_tilt.values)
k2_ss_d =stats.ks_2samp(s_cat4_s.med_tilt.values,s_cat4_d.med_tilt.values)
                  
#ad_nn_s =stats.ks_2samp(n_cat4_s.med_tilt.values,n_cat4_s.med_tilt.values)
#ad_ss_s =stats.ks_2samp(s_cat4_s.med_tilt.values,s_cat4_s.med_tilt.values)


#AD-two Plots
#ax9.text(40,0.55,'AD(NS;S) = {0:5.4f}  '.format(ad_ns_s[2]),fontsize=12)
#ax9.text(40,0.45,'AD(NS;D) = {0:5.4f}  '.format(ad_ns_d[2]),fontsize=12)
#ax9.text(40,0.35,'AD(NN;D/S) = {0:5.4f}'.format(ad_nn_d[2]),fontsize=12)
#ax9.text(40,0.25,'AD(SS;D/S) = {0:5.4f}'.format(ad_ss_d[2]),fontsize=12)
#Ks-two Plots
max_str = '$_\mathrm{M}$'
min_str = '$_\mathrm{N}$'
#Removed text from plot 2018/06/15 J. Prchlik
#ax9.text(25,0.22,'p(N{1},S{2})={0:4.3f}'.format(k2_ns_s[1],min_str,min_str),fontsize=16)
#ax9.text(25,0.17,'p(N{1},S{2})={0:4.3f}'.format(k2_ns_d[1],max_str,max_str),fontsize=16)
#ax9.text(25,0.12,'p(N{1},N{2})={0:4.3f}'.format(k2_nn_d[1],max_str,min_str),fontsize=16)
#ax9.text(25,0.07,'p(S{1},S{2})={0:4.3f}'.format(k2_ss_d[1],max_str,min_str),fontsize=16)
#ax9.text(50,0.20,'NN (S) = {0:5.4f}'.format(ad_nn_s[2]),fontsize=12)
#ax9.text(50,0.10,'SS (S) = {0:5.4f}'.format(ad_ss_s[2]),fontsize=12)
ax9.set_xlim([-90,90])


ax9.legend(loc='upper left',frameon=False,fontsize=17)
ax9.set_ylabel('Cumulative Fraction')
ax9.set_xlabel('Tilt [Deg.]')
fancy_plot(ax9)

fig9.savefig('plots/tilt_during_diff_cat4.png',bbox_pad=.1,bbox_inches='tight')
fig9.savefig('plots/tilt_during_diff_cat4.eps',bbox_pad=.1,bbox_inches='tight')


#Categories to loop over
#Just use 123 and 4 2018/05/02
#heat_map = ['allf','fil123','fil4']
#Use all filaments on 1 plot
#heat_map = ['fil123','fil4']
heat_map = ['allf','fil123','fil4']
#Tilt as a function of latitude for inbetween and during/after solar maximum times

#break down if heat_map is greater than 1 2108/05/10 J. Prchlik
if len(heat_map) > 1:
    fig_joy, axs_joy = plt.subplots(figsize=(6,len(heat_map)*6),nrows=len(heat_map),sharex=True)
else:
    fig_joy, ax_joy = plt.subplots(figsize=(6,6))

fig_joy.subplots_adjust(hspace=0.001,wspace=0.001)



#Set up bins for Joy's Law Heat map
resx = 5
resy = 5
xbins = np.arange(0,100,resx)
ybins = np.arange(0,100,resy)

#degree position for Joy's Law annotation
joy_p = 70.
#Joy's Law value
gam0 = 32.1

#loop and plot joy's law
for i,j in enumerate(heat_map):

    #add fix if making a multip plot 2018/05/10 J. Prchlik
    if len(heat_map) > 1:
        ax_joy = axs_joy[i]

    #label line as joy's law in first interation
    #Removing Joy's Law annotation 2018/06/14 J. Prchlik
    ##if i == 0: ax_joy.annotate("Joy's Law",xy=(joy_p,gam0*np.sin(np.radians(joy_p))),xycoords='data',
    ##                           xytext=(10.,90.),textcoords='data',arrowprops=dict(facecolor='black',shrink=0.0),
    ##                           horizontalalignment='left', verticalalignment='top',
    ##                           fontsize=22)

    d = fil_dict[j]
    d[0].set_index(d[0].track_id,inplace=True)
    #d[0] = d[0][~d[0].index.duplicated(keep='last')]
    #Switch to 2D histogram per Kathy's comments
    H,xedges,yedges = np.histogram2d(np.abs(d[0].med_l),np.abs(d[0].med_tilt),bins=(xbins,ybins))
    H = H.T #transpose for plotting
    #set up X,Y values
    ccmap = plt.cm.viridis.reversed()
    ccmap.set_under('1.00')
    X, Y = np.meshgrid(xedges, yedges)
    plotc = ax_joy.pcolormesh(X,Y,H,label=None,cmap=ccmap,vmin=1)
    #Add Joy's law from Stenflo & Kosovichev (2012)
    #Removing Joy's Law annotation 2018/06/14 J. Prchlik
    ##latg = np.arange(0,90)
    ##ax_joy.plot(latg,gam0*np.sin(np.radians(latg)),'--',color='black',linewidth=3)

    #Print percentage below Joy's Law 2018/05/02 J. Prchlik
    print('################################################')
    print(j)
    below_joy, = np.where(np.abs(d[0].med_tilt) < gam0*np.sin(np.radians(np.abs(d[0].med_l))))
    print(below_joy.size,len(d[0]))
    print('################################################')


    #switch to axis locator J. Prchlik
    axins = inset_axes(ax_joy,
                       width="5%",  # width = 30% of parent_bbox
                       height="40%",  # height : 1 inch
                       loc=1,borderpad=2.2)
    cbar = fig2.colorbar(plotc,cax=axins)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label('Filaments [\#]',fontsize=18)
    #ax_joy.set_title(j.upper())
    #ax_joy.legend(loc='upper right',frameon=False,scatterpoints=1)
    fancy_plot(ax_joy)
    if i == len(heat_map)-1: ax_joy.set_xlabel('$|$Latitude$|$ [Deg.]')
    #Updated to filaments with and without Cavities
    ax_joy.set_ylabel(fil_dict[j][4].replace('All Filaments','')+' $|$Tilt$|$ [Deg.]')


fig_joy.savefig('plots/filaments_joys_law.png',bbox_pad=.1,bbox_inches='tight')
fig_joy.savefig('plots/filaments_joys_law.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig_joy)


#Tilt as a function of latitude for inbetween and during/after solar maximum times
#Catagory 4 only
fig_joy, axs_joy = plt.subplots(figsize=(16,16),nrows=2,ncols=2,sharex=True,sharey=True)
axs_joy = axs_joy.ravel()
fig_joy.subplots_adjust(hspace=0.001,wspace=0.001)
#Categories to loop over
heat_map = [n_cat4_d,n_cat4_s,s_cat4_d,s_cat4_s]
heat_cap = ['NB','NA','SB','SA']

for i,d in enumerate(heat_map):
    ax_joy = axs_joy[i]
    #plot all filaments broken up by group
    #Switch to 2D histogram per Kathy's comments
    H,xedges,yedges = np.histogram2d(np.abs(d.med_l),np.abs(d.med_tilt),bins=(xbins,ybins))
    H = H.T #transpose for plotting
    #set up X,Y values
    ccmap = plt.cm.viridis.reversed()
    ccmap.set_under('1.00')
    X, Y = np.meshgrid(xedges, yedges)
    plotc = ax_joy.pcolormesh(X,Y,H,label=None,cmap=ccmap,vmin=1,vmax=11)
    #Add Joy's law from Stenflo & Kosovichev (2012)
    gam0 = 32.1
    latg = np.arange(0,90)
    #plot heatmap
    ax_joy.plot(latg,gam0*np.sin(np.radians(latg)),'--',color='black',linewidth=3)
    #add text label
    ax_joy.text(65,20,heat_cap[i])
    #switch to axis locator J. Prchlik
    axins = inset_axes(ax_joy,
                       width="5%",  # width = 30% of parent_bbox
                       height="40%",  # height : 1 inch
                       loc=1,borderpad=4.0)
    cbar = fig2.colorbar(plotc,cax=axins)
    cbar.set_label('Filaments [\#]',fontsize=18)
    fancy_plot(ax_joy)
    if i >= len(heat_map)-2: ax_joy.set_xlabel('Latitude [Deg.]')
    if ((i == 0) | (i == 2)): ax_joy.set_ylabel('$|$Tilt$|$ [Deg.]')

fig_joy.savefig('plots/filaments_wo_cavity_joys_law.png',bbox_pad=.1,bbox_inches='tight')
fig_joy.savefig('plots/filaments_wo_cavity_joys_law.eps',bbox_pad=.1,bbox_inches='tight')
plt.close(fig_joy)

#Correlation or Anti-correlation in filament tilt for category 4
# (2017/12/18 J. Prchlik)
fig_at, ax_at = plt.subplots(nrows=2,figsize=(6,12))
#fig_at.subplots_adjust(wspace=0.001,hspace=0.001)
#ax_at[0].errorbar(mbs.med_tilt_mean,mbn.med_tilt_mean,xerr=tot_err_s,yerr=tot_err_n,capsize=3,barsabove=True,fmt='o',color='black')
#ax_at[1].errorbar(mbs.med_tilt_mean,mbn.med_tilt_mean.abs(),xerr=tot_err_s,yerr=tot_err_n,capsize=3,barsabove=True,fmt='o',color='black')
ax_at[0].scatter(mbs.med_tilt_mean,mbn.med_tilt_mean,color='black')
ax_at[1].scatter(mbs.med_tilt_mean.abs(),mbn.med_tilt_mean.abs(),color='black')

#overplot when tilt are statically different
ax_at[0].scatter(mbs.loc[((mbs.index <= e_d1) | (mbs.index >= s_d2)),'med_tilt_mean']
                ,mbn.loc[((mbn.index <= e_d1) | (mbn.index >= s_d2)),'med_tilt_mean'],color='red')
ax_at[1].scatter(mbs.loc[((mbs.index <= e_d1) | (mbs.index >= s_d2)),'med_tilt_mean'].abs()
                ,mbn.loc[((mbn.index <= e_d1) | (mbn.index >= s_d2)),'med_tilt_mean'].abs(),color='red')

ax_at[0].set_xlabel('Mean Southern Tilt [Deg.]')
ax_at[1].set_xlabel('$|$Mean$|$ Southern Tilt [Deg.]')
ax_at[0].set_ylabel('Mean Northern Tilt [Deg.]')
ax_at[1].set_ylabel('$|$Mean$|$ Northern Tilt [Deg.]')


#add correlation coeoff
x1 = mbs.med_tilt_mean.values
x2 = mbs.med_tilt_mean.abs().values
y1 = mbn.med_tilt_mean.values
y2 = mbn.med_tilt_mean.abs().values
use, = np.where((np.isfinite(x1)) & (np.isfinite(y1)))
r_tilt = stats.pearsonr(x1[use],y1[use])
r_atilt = stats.pearsonr(x2[use],y2[use])

#Add correlation to output 2017/12/18 J. Prchlik
ax_at[0].text(-40.,60.,'r={0:4.3f}'.format(*r_tilt),fontsize=16,color='black')
ax_at[1].text(-40.,60.,'r={0:4.3f}'.format(*r_atilt),fontsize=16,color='black')


#add correlation coeoff when tilts are different
x1 = mbs.loc[((mbs.index <= e_d1) | (mbs.index >= s_d2)),'med_tilt_mean'].values
x2 = mbs.loc[((mbn.index <= e_d1) | (mbn.index >= s_d2)),'med_tilt_mean'].abs().values
y1 = mbn.loc[((mbs.index <= e_d1) | (mbs.index >= s_d2)),'med_tilt_mean'].values
y2 = mbn.loc[((mbn.index <= e_d1) | (mbn.index >= s_d2)),'med_tilt_mean'].abs().values
use, = np.where((np.isfinite(x1)) & (np.isfinite(y1)))
r_tilt = stats.pearsonr(x1[use],y1[use])
r_atilt = stats.pearsonr(x2[use],y2[use])


#Add correlation to output 2017/12/18 J. Prchlik
ax_at[0].text(-40.,50.,'r={0:4.3f}'.format(*r_tilt),fontsize=16,color='red')
ax_at[1].text(-40.,50.,'r={0:4.3f}'.format(*r_atilt),fontsize=16,color='red')

fancy_plot(ax_at[0])
fancy_plot(ax_at[1])

y_lim = [-50,85]
ax_at[0].set_xlim(y_lim)
ax_at[1].set_xlim(y_lim)
ax_at[0].set_ylim(y_lim)
ax_at[1].set_ylim(y_lim)


fig_at.savefig('plots/fi_tilt_ns_comp.png',bbox_pad=.1,bbox_inches='tight')
fig_at.savefig('plots/fi_tilt_ns_comp.eps',bbox_pad=.1,bbox_inches='tight')

plt.close(fig_at)

########################################################################################

#Add number of eruptions to output
fi_er = pd.read_pickle('filament_eruptions/query_output/all_fe_20120101-20141130.pic')
fi_er['events'] = 1
#get only one instance per event
fi_er.sort_index(inplace=True)
fi_er = fi_er[~fi_er.index.duplicated(keep='first')]


#cut to eruptions only above 30 degrees latitude 
n_er = fi_er[fi_er.hgs_y > 30.]
s_er = fi_er[fi_er.hgs_y < -30.]


#bin up in 4W bins 
#bn_er = n_er.resample(sam).sum()
#bs_er = s_er.resample(sam).sum()
bn_er = real_resamp(n_er,rng,col='events')
bs_er = real_resamp(s_er,rng,col='events')

#plot run N/S total 
#Switch filament eruptions to sunspot number 2017/10/13
#ax3[3].errorbar(bn_er.index,bn_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='o',color='red',label='Northern ({0})'.format(sam))
#ax3[3].errorbar(bs_er.index,bs_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='D',color='black',label='Southern ({0})'.format(sam))

#over plot Patric McCuellys Filament Eruption Catalog
pm_er = pd.read_table('catalog_table.txt',delim_whitespace=True,skiprows=27)
#remove fill values
pm_er = pm_er[((pm_er.START != '-') & (pm_er.TYPE != 'AR'))]
pm_er['time_dt'] = pd.to_datetime(pm_er.START)

#set index to be time
pm_er.set_index(pm_er['time_dt'],inplace=True)
#remove any duplicates
pm_er = pm_er[~pm_er.index.duplicated(keep='first')]
pm_er['events'] = 1  #used for counting events

#Convert 30 degrees latitude to arcseconds
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from datetime import datetime
import astropy.units as u

#line to cut the filament erutions 
cut = SkyCoord(0.*u.deg,30.*u.deg,frame=frames.HeliographicStonyhurst,obstime=datetime(2013,6,6,0,0,0))
#convert to HPC coordiantes
cuty = cut.helioprojective.Ty.value #~475-485 over a year (using 475)

#separate into north and south
pm_n = pm_er[pm_er.Y.astype('float') > cuty]
pm_s = pm_er[pm_er.Y.astype('float') <-cuty]
#get N/S sampled eruptions
pn_er = real_resamp(pm_n,rng,col='events')
ps_er = real_resamp(pm_s,rng,col='events')

#plot run N/S total 
#Switch filament eruptions to sunspot number 2017/10/13
#ax3[3].errorbar(pn_er.index,pn_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='s',color='purple',label='Northern ({0})'.format(sam))
#ax3[3].errorbar(ps_er.index,ps_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='^',color='teal',label='Southern ({0})'.format(sam))


#Switch filament eruptions to sunspot number 2017/10/13
#ax3[3].set_ylabel('Number of Eruptions')
ax3[plot_rows-1].set_ylabel('Ave. Sunspot [\#]')
fancy_plot(ax3[plot_rows-1])




#ax[1].set_yticklabels([])
#ax2[1].set_yticklabels([])
#ax2[3].set_yticklabels([])
#ax5[1].set_yticklabels([])

ax[0].set_title('Northern')
ax[1].set_title('Southern')
ax5[0].set_title('Northern')
ax5[1].set_title('Southern')
#ax4[0].set_title('Northern')
#ax4[1].set_title('Southern')

ax[0].set_xlabel('Med. Tilt [Deg.]')
ax[1].set_xlabel('Med. Tilt [Deg.]')
ax1.set_xlabel("Med. Centroid Lat. [Deg.]")
ax3[2].set_xlabel("Time [UTC]")
ax4[0].set_xlabel('Med. Tilt [Deg.]')
ax4[1].set_xlabel('Med. Tilt [Deg.]')
ax5[0].set_xlabel('Med. Tilt [Deg.]')
ax5[1].set_xlabel('Med. Tilt [Deg.]')

ax[0].set_ylabel('Cumulative Fraction')
#ax[1].set_ylabel('Cumulative Fraction',fontsize=24)
ax1.set_ylabel('Med. Tilt [Deg.]')
ax1.set_ylabel('Tilt [Deg.]')
ax2[0].set_ylabel('Cumulative Fraction')
#ax2[1].set_ylabel('Cumulative Fraction',fontsize=24)
ax4[0].set_ylabel("Med. Centroid Lat. [Deg.]")
ax4[1].set_ylabel("Med. Centroid Lat. [Deg.]")
ax5[0].set_ylabel('Cumulative Fraction')


fancy_plot(ax[0])
fancy_plot(ax[1])
fancy_plot(ax1)
fancy_plot(ax4[0])
fancy_plot(ax4[1])
fancy_plot(ax5[0])
fancy_plot(ax5[1])
#fancy_plot(ax6)
#fancy_plot(ax7)


ax[0].legend(loc='upper left',frameon=False,fontsize=16)
ax1.legend(loc='upper center',frameon=True ,handletextpad=-.112,scatterpoints=1,fontsize=18)
ax2[0].legend(loc='upper left',frameon=False,fontsize=16)
ax3[0].legend(loc='upper left',frameon=False,handletextpad=.112,scatterpoints=1,fontsize=12,handlelength=1)
ax4[0].legend(loc='lower right',frameon=False,fontsize=18)
ax5[0].legend(loc='upper left',frameon=False,fontsize=18)
ax6.legend(loc='lower right',scatterpoints=1,handletextpad=-0.112,frameon=False,fontsize=12)

#set plot range sunsplot pot
ax3[0].set_xlim([datetime(2011,1,1),datetime(2015,2,1)])

fig.savefig( 'plots/ns_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig1.savefig('plots/med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig2.savefig('plots/ns_cat_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig3.savefig('plots/tilt_v_time_w_ss.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig4.savefig('plots/ns_med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig5.savefig('plots/ns_cumla_dis_tilt_comb12.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig.savefig( 'plots/ns_cumla_dis_tilt.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig1.savefig('plots/med_tilt_v_med_lat.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig2.savefig('plots/ns_cat_cumla_dis_tilt.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig3.savefig('plots/tilt_v_time_w_ss.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig4.savefig('plots/ns_med_tilt_v_med_lat.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig5.savefig('plots/ns_cumla_dis_tilt_comb12.eps',bbox_pad=.1,bbox_inches='tight',fontsize=18)
#fig6.savefig('plots/ns_tilt_ss_height_fil4.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
#fig7.savefig('plots/ns_diff_tilt_diff_ss_height_fil4.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)