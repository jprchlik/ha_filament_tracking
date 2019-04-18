import datetime 
import numpy as np

class add_props:
    """
    add_props adds useful properties to FITracked pandas object.

    add_props adds a few useful properties to the FITracked pandas object.
    The list of added properties are as follows:
    two date time arrays for the track observed start date and end date,
    the mean value of the Polygon,
    and the start and end date of the entire track.
    
    Parameters
    ----------
    None

    Returns
    -------
    None


    """

    def __init__(self,dat):
        """
        Init and add all properties to pandas object

        """
        self.dat = dat
        self.add_dt_obj()
        self.add_mean_val()
        self.add_track_span()

#add datetime object to array
    def add_dt_obj(self):
        """
        Add event start and end time datetime objects

        """

        #date format
        dfmt = '%Y-%m-%dT%H:%M:%S'
        
        #create a date time array
        dtstr = ['start','end']
        for j in dtstr:
            self.dat['event_{0}time_dt'.format(j)] = [ datetime.datetime.strptime(i,dfmt) for i in self.dat['event_{0}time'.format(j)]]
    
#add mean values to array
    def add_mean_val(self):
        vals =  [ calc_mean_pos(coor) for coor in self.dat['hpc_boundcc']]
        vals = np.array(vals)
        
        self.dat['meanx'] =vals[:,0]
        self.dat['meany'] =vals[:,1]

#find full track end and start time
    def add_track_span(self):
        tid = np.unique(self.dat['track_id'].values)
        self.dat['total_event_start'] = 0
        self.dat['total_event_end']   = 0
        #Switched to faster grouping 2019/04/18 J. Prchlik
        start = self.dat.groupby('track_id').event_starttime_dt.min()
        end   = self.dat.groupby('track_id').event_endtime_dt.min()
        start.rename('total_event_start',inplace=True)
        end.rename('total_event_end',inplace=True)
        self.dat = self.dat.set_index('track_id').join(end).join(start)
        self.dat.reset_index(inplace=True)
        #for k in tid:
        #    track, = np.where(self.dat['track_id'].values == k)
        #    self.dat['total_event_start'][track] = np.min(self.dat['event_starttime_dt'][track])
        #    self.dat['total_event_end'][track] = np.max(self.dat['event_endtime_dt'][track])


    
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

