#add_properties.py
is a class that adds the mean value, observed start and end time as datetime objects, and a track long start and end time as datetime objects. 

#create_plots.py 
is a class that plots all tracks with a start and end time spanning the observed H alpha time. It also predicts solar rotation to project where the track should be based on the values from sunpy.physics solar_rotation.

#grab_gong.py
is a class that downloads archived GONG data to a local directory structure.

#make_gong_cat.py
is a script which downloads archived GONG H alpha data, processes it, overplots h alpha filament tracks, and creates a movie from the observations.


#make_movie.py
is a class which creates a movie from files of a given extension (default ='png').
