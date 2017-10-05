import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 24



import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
import pandas as pd
import numpy as np
from datetime import timedelta

import scipy.stats as stats
import statsmodels.api as sm

def setup_dis(x,col='med_tilt'):
    x.set_index(x['track_id'],inplace='true')
    x.sort_values(by=col,inplace=True)
    x[len(x)] = x.iloc[-1]
    x['dis']  = np.linspace(0.,1.,len(x))

    #add value so all distributions start on Jan. 1st
    x.loc[-1] = [np.nan]*x.shape[1]
    return x


def real_resamp(x,dates,col='med_tilt'):

    y = pd.DataFrame(index=dates)
    y[col+'_mean'] = np.nan
    y[col+'_med'] = np.nan
    y[col+'_std'] = np.nan
    y[col+'_sum'] = np.nan
    y[col+'_cnt'] = np.nan
   
    #total number of dates
    t = len(dates)

    #return y

    for j,i in enumerate(dates):

        if j < t-2:
            use, = np.where((x.index > i) & (x.index < dates[j+1]))
        else:
            use, = np.where(x.index > i)
          
        if use.size > 0:
            y.loc[i,col+'_mean'] = np.mean(x[col].values[use])
            y.loc[i,col+'_med']  = np.median(x[col].values[use])
            y.loc[i,col+'_std']  = np.std(x[col].values[use])
            y.loc[i,col+'_sum']  = np.sum(x[col].values[use])
            y.loc[i,col+'_cnt']  = use.size

    #Add time average time offset to days
    #toff = x.index[1:]-x.index[:-1]
    #x.index = x.index+toff/2.
    y.index = y.index+pd.DateOffset(days=14)

    return y



#sampling frequency 
sam = '4W'
#get pandas timeseries representation for filament tracking code time range
rng = pd.date_range('2012-01-01 00:00:00','2015-01-01 00:00:00',freq=sam)#.to_timestamp()

#read in filament categories given in Brianna's code
fil = pd.read_pickle('filament_catagories.pic')

fil_dict = {}
fil_fmt = 'fil{0:1d}'

fil_keys = ['fil1','fil2','fil3','fil4']

fil_dict['fil1'] = [fil[fil.cat_id == 1],'red'  ,'o','-' ,"Cat. 1"]
fil_dict['fil2'] = [fil[fil.cat_id == 2],'black','x','--',"Cat. 2"]
fil_dict['fil12'] = [fil[((fil.cat_id == 1) | (fil.cat_id == 2))],'red'  ,'o','-' ,"Cat. 1 and 2"]
fil_dict['fil3'] = [fil[fil.cat_id == 3],'teal' ,'s','-.',"Cat. 3"]
fil_dict['fil4'] = [fil[fil.cat_id == 4],'blue' ,'D',':' ,"Cat. 4"]
fil_dict['allf'] = [fil[fil.cat_id != 0],'blue' ,'D',':' ,"All Filaments"]


fig, ax = plt.subplots(ncols=2,figsize=(11,8.5))
fig1, ax1 = plt.subplots(figsize=(11.,8.5))
fig2, ax2 = plt.subplots(figsize=(13.,17.),ncols=2,nrows=2)
fig4, ax4 = plt.subplots(nrows=2,figsize=(8.5,11),sharex=True)
fig5, ax5 = plt.subplots(ncols=2,figsize=(11,8.5))
ax2 = ax2.ravel()
fig.subplots_adjust(hspace=0.001,wspace=0.001)
fig2.subplots_adjust(wspace=0.001)
fig4.subplots_adjust(hspace=0.001)
fig5.subplots_adjust(hspace=0.001,wspace=0.001)


#compare stable vs unstable filaments
stab_keys = ['fil1','fil2','fil12','fil3','fil4','allf']
for i in stab_keys: 
    d = fil_dict[i]
    d[0]['north'] = 0
    d[0]['north'][d[0].med_y > 0.] = 1



for j,i in enumerate(fil_keys):
    d = fil_dict[i]

    d[0].set_index(d[0]['track_id'],inplace=True)
    d[0] = d[0][~d[0].index.duplicated(keep='first')]

    d[0].set_index(d[0]['event_starttime_dt'],inplace=True)
    d[0].sort_index(inplace=True)

    n = d[0][d[0].north == 1]
    s = d[0][d[0].north == 0]

    n = setup_dis(n)
    s = setup_dis(s)
    n.set_index(n['event_starttime_dt'],inplace=True)
    n.sort_index(inplace=True)
    s.set_index(s['event_starttime_dt'],inplace=True)
    s.sort_index(inplace=True)


    #two sample anderson-darling test between n and s of same catagory 
    ad = stats.anderson_ksamp([n.med_tilt.values,s.med_tilt.values])
    k2 = stats.ks_2samp(s.med_tilt.values,n.med_tilt.values)
    
   
    ax[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4])
    ax[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4])


    ax4[0].scatter(n.med_tilt,n.med_y,color=d[1],marker=d[2],label=d[4])
    ax4[1].scatter(s.med_tilt,s.med_y,color=d[1],marker=d[2],label=d[4])

    ax1.scatter(d[0].med_y,d[0].med_tilt,color=d[1],marker=d[2],label=d[4])

    ax2[j].plot(n.med_tilt,n.dis,color='red',label='Nothern')
    ax2[j].plot(s.med_tilt,s.dis,color='black',linestyle='--',label='Southern')
    if ad[-1] < 1.0: ax2[j].text(20,.1,'p(A-D) = {0:5.4f}'.format(ad[-1]),fontsize=18)
    ax2[j].text(20,.15,'p(KS2) = {0:5.4f}'.format(k2[-1]),fontsize=18)
    ax2[j].set_title(d[4])
    ax2[j].set_xlabel('Med. Tilt [Deg.]')
    ax2[j].set_xlim([-95,95])
    fancy_plot(ax2[j])


    #do the camparision for stable filaments vs no stable
    if i == 'fil1':
        d = fil_dict['fil12']
        d[0] = d[0][~d[0].index.duplicated(keep='first')]

        n = d[0][d[0].north == 1]
        s = d[0][d[0].north == 0]
        n = setup_dis(n)
        s = setup_dis(s)

        ax5[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4])


        #setup d3 and d4 distributions for comparision
        d3 = fil_dict['fil3'][0]
        d3.set_index(d3['track_id'],inplace='true')
        d3.sort_index(inplace=True)
        d3 = d3[~d3.index.duplicated(keep='first')]

        d4 = fil_dict['fil4'][0]
        d4.set_index(d4['track_id'],inplace='true')
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

        #show fit stat on plot
        if ad[-1] < 1.0: ax5[0].text(5,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3n[-1]),fontsize=14)
        ax5[0].text(5,.15,'p(KS2;12,3) = {0:5.4f}'.format(k23n[-1]),fontsize=14)
        if ad[-1] < 1.0: ax5[1].text(5,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3s[-1]),fontsize=14)
        ax5[1].text(5,.15,'p(KS2;12,3) = {0:5.4f}'.format(k23s[-1]),fontsize=14)


    elif ((i == 'fil3') | (i == 'fil4')):
        ax5[0].plot(n.med_tilt,n.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[1].plot(s.med_tilt,s.dis,color=d[1],linestyle=d[3],label=d[4])




#get person r value for ax4[0] (north) and ax4[1] (south)
allf = fil_dict['allf'][0]
allf.set_index(allf['event_starttime_dt'],inplace=True)
allf.sort_index(inplace=True)
   

npp = stats.pearsonr(allf[allf.north == 1].med_tilt.values,allf[allf.north == 1].med_y.values)
spp = stats.pearsonr(allf[allf.north == 0].med_tilt.values,allf[allf.north == 0].med_y.values)
ax4[0].text(-90,200,'r={0:4.3f},p={0:4.3f}'.format(*npp),fontsize=12)
ax4[1].text(-90,-200,'r={0:4.3f},p={0:4.3f}'.format(*spp),fontsize=12)

#plotting 1and 2, 3, and 4 versus time
fig3, ax3 = plt.subplots(figsize=(33.,34.0),nrows=4,sharex=True)
fig3.subplots_adjust(hspace=0.001,wspace=0.001)

#array of filament objects
tilt_time = ['fil12','fil3','fil4']

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
    
    #plot running mean
    ax3[j].errorbar(mbn.index,mbn.med_tilt_mean,xerr=timedelta(days=14),yerr=mbn.med_tilt_std.values/np.sqrt(mbn.med_tilt_cnt.values),capsize=3,barsabove=True,fmt='-',color='red',linewidth=3,label='Northern Mean ({0})'.format(sam))
    ax3[j].errorbar(mbs.index,mbs.med_tilt_mean,xerr=timedelta(days=14),yerr=mbs.med_tilt_std.values/np.sqrt(mbs.med_tilt_cnt.values),capsize=3,barsabove=True,fmt='--',color='black',linewidth=3,label='Southern Mean ({0})'.format(sam))
    
    #Make tilt versus time plot
    ax3[j].scatter(bn.index,bn.med_tilt,color='red',marker='o',label='Northern')
    ax3[j].scatter(bs.index,bs.med_tilt,color='black',marker='D',label='Southern')

    #Y title
    ax3[j].set_ylabel("Med. Tilt [Deg.]\r {0}".format(i.replace('fil','Category ').replace('12','1 and 2')))
    fancy_plot(ax3[j])
    ax3[j].set_ylim([-90.,90.])

#Add number of eruptions to output
fi_er = pd.read_pickle('filament_eruptions/query_output/all_fe_20120101-20141130.pic')
fi_er['events'] = 1
#get only one instance per event
fi_er.sort_index(inplace=True)
fi_er = fi_er[~fi_er.index.duplicated(keep='first')]


#cut to eruptions only above 30 degrees latitude 
n_er = fi_er[fi_er.hgs_y >  30.]
s_er = fi_er[fi_er.hgs_y < -30.]



#bin up in 4W bins 
#bn_er = n_er.resample(sam).sum()
#bs_er = s_er.resample(sam).sum()
bn_er = real_resamp(n_er,rng,col='events')
bs_er = real_resamp(s_er,rng,col='events')

#plot run N/S total 
ax3[3].errorbar(bn_er.index,bn_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='o',color='red',label='Northern ({0})'.format(sam))
ax3[3].errorbar(bs_er.index,bs_er.events_sum,xerr=timedelta(days=14),capsize=3,barsabove=True,fmt='D',color='black',label='Southern ({0})'.format(sam))
ax3[3].set_ylabel('Number of Eruptions')
fancy_plot(ax3[3])



ax[1].set_yticklabels([])
ax2[1].set_yticklabels([])
ax2[3].set_yticklabels([])
ax5[1].set_yticklabels([])

ax[0].set_title('Northern')
ax[1].set_title('Southern')
ax5[0].set_title('Northern')
ax5[1].set_title('Southern')
#ax4[0].set_title('Northern')
#ax4[1].set_title('Southern')

ax[0].set_xlabel('Med. Tilt [Deg.]')
ax[1].set_xlabel('Med. Tilt [Deg.]')
ax1.set_xlabel("Med. Centroid Lat. ['']")
ax3[2].set_xlabel("Time")
ax4[0].set_xlabel('Med. Tilt [Deg.]')
ax4[1].set_xlabel('Med. Tilt [Deg.]')
ax5[0].set_xlabel('Med. Tilt [Deg.]')
ax5[1].set_xlabel('Med. Tilt [Deg.]')

ax[0].set_ylabel('Cumulative Fraction')
ax1.set_ylabel('Med. Tilt [Deg.]')
ax1.set_ylabel('Tilt [Deg.]')
ax2[0].set_ylabel('Cumulative Fraction')
ax2[2].set_ylabel('Cumulative Fraction')
ax4[0].set_ylabel("Med. Centroid Lat. ['']")
ax4[1].set_ylabel("Med. Centroid Lat. ['']")
ax5[0].set_ylabel('Cumulative Fraction')


fancy_plot(ax[0])
fancy_plot(ax[1])
fancy_plot(ax1)
fancy_plot(ax4[0])
fancy_plot(ax4[1])
fancy_plot(ax5[0])
fancy_plot(ax5[1])


ax[0].legend(loc='upper left',frameon=False,fontsize=18)
ax1.legend(loc='upper center',frameon=True ,handletextpad=-.112,scatterpoints=1,fontsize=18)
ax2[0].legend(loc='upper left',frameon=False,fontsize=18)
ax3[0].legend(loc='upper left',frameon=False,handletextpad=.112,scatterpoints=1,fontsize=18,handlelength=1)
ax4[0].legend(loc='lower right',frameon=False,fontsize=18)
ax5[0].legend(loc='upper left',frameon=False,fontsize=18)

fig.savefig( 'plots/ns_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig1.savefig('plots/med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig2.savefig('plots/ns_cat_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig3.savefig('plots/tilt_v_time_w_fe.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig4.savefig('plots/ns_med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig5.savefig('plots/ns_cumla_dis_tilt_comb12.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)