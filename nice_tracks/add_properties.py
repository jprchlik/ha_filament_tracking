import datetime 
import numpy as np

class add_props():

    def __init__(self,dat):
        self.dat = dat
        self.add_dt_obj()
        self.add_mean_val()

#add datetime object to array
    def add_dt_obj(self):
        #date format
        dfmt = '%Y-%m-%dT%H:%M:%S'
        
        #create a date time array
        dtstr = ['start','end']
        for j in dtstr:
            self.dat['event_{0}time_dt'.format(j)] = [ datetime.datetime.strptime(i,dfmt) for i in self.dat['event_{0}time'.format(j)]]
    
#add mean values to array
    def add_mean_val(self):
        vals =  [ calc_mean_pos(coor) for coor in self.dat['hpc_bbox']]
        vals = np.array(vals)
        
        self.dat['meanx'] =vals[:,0]
        self.dat['meany'] =vals[:,1]


    
def calc_mean_pos(coor):
#list of polygon coordinates
    corlist =  coor.replace('POLYGON((','').replace('))','').split(',')
#create array and calculate mean x,y value for start
    corarra =  np.zeros((len(corlist),2))
    for j,k in enumerate(corlist):
        values = k.split(' ')
        corarra[j,0] = float(values[0])
        corarra[j,1] = float(values[1])

#find mean x and y values
    meanx, meany = np.mean(corarra[:,0]),np.mean(corarra[:,1])

    return meanx, meany

