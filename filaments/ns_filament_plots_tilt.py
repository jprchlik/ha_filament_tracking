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
    return x


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
fig3, ax3 = plt.subplots(figsize=(33.,8.5))
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

    n = d[0][d[0].north == 1]
    s = d[0][d[0].north == 0]

    n = setup_dis(n)
    s = setup_dis(s)
    n.set_index(n['event_starttime_dt'],inplace=True)
    s.set_index(s['event_starttime_dt'],inplace=True)


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
allf = fil_dict['fil4'][0]
allf.set_index(allf['event_starttime_dt'],inplace=True)
bn = allf[allf.north == 1]
bs = allf[allf.north == 0]
   

npp = stats.pearsonr(allf[allf.north == 1].med_tilt.values,allf[allf.north == 1].med_y.values)
spp = stats.pearsonr(allf[allf.north == 0].med_tilt.values,allf[allf.north == 0].med_y.values)
ax4[0].text(-90,200,'r={0:4.3f},p={0:4.3f}'.format(*npp),fontsize=12)
ax4[1].text(-90,-200,'r={0:4.3f},p={0:4.3f}'.format(*spp),fontsize=12)


#http://benalexkeen.com/resampling-time-series-data-with-pandas/
#get running mean
mbn = bn.resample('4W').mean()
mbs = bs.resample('4W').mean()
#get running mean
sbn = bn.resample('4W').std()
sbs = bs.resample('4W').std()
#get running count
cbn = bn.resample('4W').count()
cbs = bs.resample('4W').count()

#plot running mean
ax3.errorbar(mbn.index,mbn.med_tilt,xerr=timedelta(days=14),yerr=sbn.med_tilt.values/np.sqrt(cbn.med_tilt.values),capsize=3,barsabove=True,fmt='-',color='red',linewidth=3,label='Northern Mean (4W)')
ax3.errorbar(mbs.index,mbs.med_tilt,xerr=timedelta(days=14),yerr=sbs.med_tilt.values/np.sqrt(cbs.med_tilt.values),capsize=3,barsabove=True,fmt='--',color='black',linewidth=3,label='Southern Mean (4W)')

#Make tilt versus time plot
ax3.scatter(bn.index,bn.med_tilt,color='red',marker='o',label='Northern')
ax3.scatter(bs.index,bs.med_tilt,color='black',marker='D',label='Southern')

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
ax3.set_xlabel("Time")
ax4[0].set_xlabel('Med. Tilt [Deg.]')
ax4[1].set_xlabel('Med. Tilt [Deg.]')
ax5[0].set_xlabel('Med. Tilt [Deg.]')
ax5[1].set_xlabel('Med. Tilt [Deg.]')

ax[0].set_ylabel('Cumulative Fraction')
ax1.set_ylabel('Med. Tilt [Deg.]')
ax1.set_ylabel('Tilt [Deg.]')
ax2[0].set_ylabel('Cumulative Fraction')
ax2[2].set_ylabel('Cumulative Fraction')
ax3.set_ylabel("Med. Tilt [Deg.]")
ax4[0].set_ylabel("Med. Centroid Lat. ['']")
ax4[1].set_ylabel("Med. Centroid Lat. ['']")
ax5[0].set_ylabel('Cumulative Fraction')


fancy_plot(ax[0])
fancy_plot(ax[1])
fancy_plot(ax1)
fancy_plot(ax3)
fancy_plot(ax4[0])
fancy_plot(ax4[1])
fancy_plot(ax5[0])
fancy_plot(ax5[1])


ax[0].legend(loc='upper left',frameon=False,fontsize=18)
ax1.legend(loc='upper center',frameon=True ,handletextpad=-.112,scatterpoints=1,fontsize=18)
ax2[0].legend(loc='upper left',frameon=False,fontsize=18)
ax3.legend(loc='upper left',frameon=False,handletextpad=.112,scatterpoints=1,fontsize=18,handlelength=1)
ax4[0].legend(loc='lower right',frameon=False,fontsize=18)
ax5[0].legend(loc='upper left',frameon=False,fontsize=18)

fig.savefig( 'plots/ns_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig1.savefig('plots/med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig2.savefig('plots/ns_cat_cumla_dis_tilt.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig3.savefig('plots/tilt_v_time.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig4.savefig('plots/ns_med_tilt_v_med_lat.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)
fig5.savefig('plots/ns_cumla_dis_tilt_comb12.png',bbox_pad=.1,bbox_inches='tight',fontsize=18)