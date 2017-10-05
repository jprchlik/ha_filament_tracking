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
from datetime import datetime,timedelta

import scipy.stats as stats
import statsmodels.api as sm

def setup_dis(x,col='med_y'):
    x.set_index(x['track_id'],inplace='true')
    x.sort_values(by=col,inplace=True)
    x[len(x)] = x.iloc[-1]
    x['dis']  = np.linspace(0.,1.,len(x))
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

fil = pd.read_pickle('filament_catagories.pic')

fil_dict = {}
fil_fmt = 'fil{0:1d}'

fil_keys = ['fil1','fil2','fil3','fil4']

fil_dict['fil1'] = [fil[fil.cat_id == 1],'red'  ,'o','-' ,"Cat. 1"]
fil_dict['fil2'] = [fil[fil.cat_id == 2],'black','x','--',"Cat. 2"]
fil_dict['fil12'] = [fil[((fil.cat_id == 1) | (fil.cat_id == 2))],'red'  ,'o','-' ,"Cat. 1 and 2"]
fil_dict['fil3'] = [fil[fil.cat_id == 3],'teal' ,'s','-.',"Cat. 3"]
fil_dict['fil4'] = [fil[fil.cat_id == 4],'blue' ,'D',':' ,"Cat. 4"]
fil_dict['allf'] = [fil[fil.cat_id != 0],'blue' ,'D',':' ,"Cat. 4"]
fil_dict['alll'] = [fil,'blue' ,'D',':' ,"Cat. 4"]


fig, ax = plt.subplots(ncols=2,figsize=(11,8.5))
fig2, ax2 = plt.subplots(figsize=(13.,17.),ncols=2,nrows=2)
#fig3, ax3 = plt.subplots(figsize=(33.,8.5))
fig5, ax5 = plt.subplots(ncols=2,figsize=(11,8.5))
ax2 = ax2.ravel()
fig.subplots_adjust(hspace=0.001,wspace=0.001)
fig2.subplots_adjust(wspace=0.001)
fig5.subplots_adjust(hspace=0.001,wspace=0.001)


#compare stable vs unstable filaments
stab_keys = ['fil1','fil2','fil12','fil3','fil4','allf','alll']
for i in stab_keys: 
    d = fil_dict[i]
    d[0]['north'] = 0
    d[0]['north'][d[0].med_y > 0.] = 1



for j,i in enumerate(fil_keys):
    d = fil_dict[i]

    d[0].set_index(d[0]['track_id'],inplace=True)
    d[0] = d[0][~d[0].index.duplicated(keep='first')]

    d[0].set_index(d[0]['event_starttime_dt'],inplace=True)

    n = d[0][d[0].north == 1]
    s = d[0][d[0].north == 0]

    n = setup_dis(n)
    s = setup_dis(s)


    #two sample anderson-darling test between n and s of same catagory 
    ad = stats.anderson_ksamp([n.med_y.values,-s.med_y.values])
    k2 = stats.ks_2samp(-s.med_y.values,n.med_y.values)
    
   
    ax[0].plot(n.med_y,n.dis,color=d[1],linestyle=d[3],label=d[4])
    ax[1].plot(s.med_y,s.dis,color=d[1],linestyle=d[3],label=d[4])



    #ax3.scatter(d[0].index,d[0].med_y,color=d[1],marker=d[2],label=d[4])

    ax2[j].plot(np.abs(n.med_y),n.dis,color='red',label='Nothern')
    ax2[j].plot(np.abs(s.med_y),1.-s.dis,color='black',linestyle='--',label='Southern')
    if ad[-1] < 1.0: ax2[j].text(100,.1,'p(A-D) = {0:5.4f}'.format(ad[-1]),fontsize=18)
    ax2[j].text(100,.15,'p(KS2) = {0:5.4f}'.format(k2[-1]),fontsize=18)
    ax2[j].set_title(d[4])
    ax2[j].set_xlabel(r"|Med. Latitude| ['']")
    ax2[j].set_xlim([80,900])
    fancy_plot(ax2[j])


    #do the camparision for stable filaments vs no stable
    if i == 'fil1':
        d = fil_dict['fil12']
        d[0] = d[0][~d[0].index.duplicated(keep='first')]

        n = d[0][d[0].north == 1]
        s = d[0][d[0].north == 0]
        n = setup_dis(n)
        s = setup_dis(s)

        ax5[0].plot(n.med_y,n.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[1].plot(s.med_y,s.dis,color=d[1],linestyle=d[3],label=d[4])


        #setup d3 and d4 distributions for comparision
        d3 = fil_dict['fil3'][0]
        d3.set_index(d3['track_id'],inplace='true')
        d3 = d3[~d3.index.duplicated(keep='first')]

        d4 = fil_dict['fil4'][0]
        d4.set_index(d4['track_id'],inplace='true')
        d4 = d4[~d4.index.duplicated(keep='first')]

        #break d3 and d4 into frames of north and south
        d3n = setup_dis(d3[d3['north'] == 1])
        d3s = setup_dis(d3[d3['north'] == 0])
        d4n = setup_dis(d4[d4['north'] == 1])
        d4s = setup_dis(d4[d4['north'] == 0])

        #two sample anderson-darling test between n or s of differnt catagories
        ad3n = stats.anderson_ksamp([d3n.med_y.values,n.med_y.values])
        k23n = stats.ks_2samp(d3n.med_y.values,n.med_y.values)
        ad3s = stats.anderson_ksamp([d3s.med_y.values,s.med_y.values])
        k23s = stats.ks_2samp(d3s.med_y.values,s.med_y.values)

        #show fit stat on plot
        if ad[-1] < 1.0: ax5[0].text(550,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3n[-1]),fontsize=14)
        ax5[0].text(550,.15,'p(KS2;12,3) = {0:5.4f}'.format(k23n[-1]),fontsize=14)
        if ad[-1] < 1.0: ax5[1].text(-530,.1,'p(A-D;12,3) = {0:5.4f}'.format(ad3s[-1]),fontsize=14)
        ax5[1].text(-530,.15,'p(KS2;12,3) = {0:5.4f}'.format(k23s[-1]),fontsize=14)


    elif ((i == 'fil3') | (i == 'fil4')):
        ax5[0].plot(n.med_y,n.dis,color=d[1],linestyle=d[3],label=d[4])
        ax5[1].plot(s.med_y,s.dis,color=d[1],linestyle=d[3],label=d[4])




#plotting 1and 2, 3, and 4 versus time
fig3, ax3 = plt.subplots(figsize=(33.,34.0),nrows=4,sharex=True)
fig3.subplots_adjust(hspace=0.001,wspace=0.001)

#array of filament objects
tilt_time = ['fil12','fil3','fil4','allf']

#loops over filament types
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

    #resample with fixed cadence
    mbn = real_resamp(bn,rng,col='med_y')
    mbs = real_resamp(bs,rng,col='med_y')
    
    #plot running mean
    ax3[j].errorbar(mbn.index,mbn.med_y_mean,xerr=timedelta(days=14),yerr=mbn.med_y_std.values/np.sqrt(mbn.med_y_cnt.values),capsize=3,barsabove=True,fmt='-',color='red',linewidth=3,label='Northern Mean ({0})'.format(sam))
    ax3[j].errorbar(mbs.index,-mbs.med_y_mean,xerr=timedelta(days=14),yerr=mbs.med_y_std.values/np.sqrt(mbs.med_y_cnt.values),capsize=3,barsabove=True,fmt='--',color='black',linewidth=3,label='Southern Mean ({0})'.format(sam))
    
    #Make y versus time plot
    ax3[j].scatter(bn.index,bn.med_y,color='red',marker='o',label='Northern')
    ax3[j].scatter(bs.index,-bs.med_y,color='black',marker='D',label='Southern')
    #Y title
    ax3[j].set_ylabel("Med. Lat. ['']\r {0}".format(i.replace('fil','Category ').replace('12','1 and 2').replace('allf','All')))
    fancy_plot(ax3[j])
    ax3[j].set_ylim([0.,990.])
    fancy_plot(ax3[j])



ax[1].set_yticklabels([])
ax2[1].set_yticklabels([])
ax2[3].set_yticklabels([])
ax5[1].set_yticklabels([])

ax[0].set_title('Northern')
ax[1].set_title('Southern')
ax5[0].set_title('Northern')
ax5[1].set_title('Southern')

ax[0].set_xlabel("Med. Latitude ['']")
ax[1].set_xlabel("Med. Latitude ['']")
ax3[2].set_xlabel("Time")
ax5[0].set_xlabel("Med. Latitude ['']")
ax5[1].set_xlabel("Med. Latitude ['']")

ax[0].set_ylabel('Cumulative Fraction')
ax2[0].set_ylabel('Cumulative Fraction')
ax2[2].set_ylabel('Cumulative Fraction')
#ax3.set_ylabel("Med. Latitue ['']")
ax5[0].set_ylabel('Cumulative Fraction')


fancy_plot(ax[0])
fancy_plot(ax[1])
fancy_plot(ax5[0])
fancy_plot(ax5[1])


ax[0].legend(loc='upper left',frameon=False,fontsize=18)
ax2[0].legend(loc='upper left',frameon=False,fontsize=18)
ax3[0].legend(loc='lower left',frameon=False,handletextpad=.112,scatterpoints=1,fontsize=18,handlelength=1)
ax5[0].legend(loc='upper left',frameon=False,fontsize=18)

fig.savefig( 'plots/ns_cumla_dis_medy.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig2.savefig('plots/ns_cat_cumla_dis_medy.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig3.savefig('plots/medy_v_time.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig5.savefig('plots/ns_cumla_dis_medy_comb12.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)