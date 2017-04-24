import ftplib
import sys
import calendar
from datetime import datetime,timedelta
import os
from multiprocessing import Pool
from astropy.io import fits
import numpy as np

typdic = {}
typdic['large'] = 'hag'
typdic['fits'] = 'haf'


class grab_gong:
    """The grab_gong class creates an object which can download files from GONG ftp archive. 
       It takes just a start and endtime after initialization by main """

    def __init__(self,start,end,ftp,arcdir,nproc,cad,verbose):
        """grab_gong class initialization which is called and set by main"""
        #get a list of directories
        self.dlev1 = ftp.nlst()
        self.ftp = ftp
        self.ftpdir = ftp.pwd()
        self.end = end
        self.start = start
        self.arcdir = arcdir
        self.nproc = nproc
        self.verbose = verbose
        self.cad = cad
# create local archive
        self.mkloc_arc()
#grab the ftp archive files in range
        self.getfiles()

    def mkloc_arc(self):
        """mkloc_arc creates an local file structure archive based on the available dates """
#make local subdirectories
        for i in self.dlev1: 
            self.mkdirs(i)
#create the day subdirectories
            days = calendar.monthrange(int(i[:4]),int(i[4:]))[1]
            for j in range(1,days+1): 
                ddir = '{0}/{0}{1:2d}'.format(i,j).replace(' ','0')
                self.mkdirs(ddir)

#Actually make the directories                 
    def mkdirs(self,adir):
        """Test whether a directory archive already exits. If the directory exists then continue."""
        try:
            os.mkdir(adir)
        except:
            if self.verbose:
                print 'Directory {0} already exists'.format(adir)

#retrieve desired cadence from file list
    def des_cad(self,start,end,delta):
        """Create an array from start to end with desired cadence"""
        curr = start
        while curr < end:
            yield curr
            curr += delta



#get files in date range from ftp archive
    def getfiles(self):
        """Retrieve file list and download files from the GONG archive. It requires not addition arguments than those set up by the object."""
    # list of days in range as datetime object
        dates = [ self.start +timedelta(n) for n in range(int ((self.end - self.start).days))]

    #desired cadence for the observations
        real_cad = [result for result in self.des_cad(self.start,self.end,timedelta(minutes=self.cad))]
        real_cad = np.array(real_cad) #turn into numpy array
    #create day file list
        self.filelist = []

#list for files and their observation times
        templist = []
        timelist = []


    #get all files in date range from ftp server
        for i in dates:
            fulldir = self.ftpdir+'/'+i.strftime('%Y%m/%Y%m%d')
            self.ftp.cwd(fulldir)
            inlist = self.ftp.nlst()
            for j in inlist: templist.append(j) #get list of files on server

        templist = [s for s in templist if "L" not in s] #skip Learmonth, Australia because they are not rotated properly

        timelist = [datetime.strptime(k[:14],'%Y%m%d%H%M%S') for k in templist] #file list as datetime objects
        timelist = np.array(timelist) #convert timelist to numpy array

#CONSUMES TOO MUCH MEMORY TO DO SUBTRACTION OVER YEARS
####        diff_mat = np.abs(np.matrix(real_cad).T-np.matrix(timelist)) # find the minimum time corresponding to minimum
####       
        index_list = [] #list of index to keep from ftp gong server
        for j,p in enumerate(real_cad): #loop over all cadence values to find best array values
            k = np.abs(timelist-p)
            rindex, = np.where(k == k.min()) #get the nonzero array index value
            index_list.append(rindex[0]) # add index list to call for download
####            
####
####        templist = np.array(templist) # allow index calling
        self.ftp =  ftplib.FTP('gong2.nso.edu','anonymous') #make sure you are properly connect)ed
        self.ftp.cwd('HA/haf')
        for p in index_list:
            failed = True # check that file passed quick quality check (sharpness greater than .01 empirically determined)
            m = p
        
            k = 0
            while failed:
                p = m+k #jump back and forth until a file passes inspection 
                j = templist[m] # get the index corresponding to the closest time
                i = timelist[m]
#change local directory
                os.chdir(self.arcdir+'/'+i.strftime('%Y%m/%Y%m%d'))
#change ftp directory
                fulldir = self.ftpdir+'/'+i.strftime('%Y%m/%Y%m%d')
                if self.ftp.pwd() != fulldir: self.ftp.cwd(fulldir)


                failed = self.write_loc_files(j)
                k = np.abs(k)+1
                k = k*(-1)**k #alt looking before and after index for a good file
                if abs(k) > 10: #max out out on 5 each way
                    failed = False
                    i = timelist[p]
                    j = templist[p]
                if failed == False: #only add to process file list (i.e. movie list) if time passes quality checks
                    self.filelist.append(self.arcdir+'/'+i.strftime('%Y%m/%Y%m%d')+'/'+j)
 
    #move back to parent directory
    #retrieve files from server
#            pool = Pool(processes=self.nproc)
#            out = pool.map(self.write_loc_files,templist)
#            pool.close()
        
        
        
    def write_loc_files(self,fname):
        """Check to see if a local version of the files already exists. Skip if it exits, but download if it does not."""
   
#check to see if file exists 
        testfile = os.path.isfile(fname)
        

#if file does not exist 
        if testfile == False:
            fhandle = open(fname,'wb')    
            try:
                self.ftp.retrbinary('RETR {0}'.format(fname),fhandle.write)
            except:#prevents file permission error from killing loop
                fhandle.close()
                return True
          
                
            fhandle.close()

#data data to roughly see if it is good
        try:
            dat = fits.open(fname)
            failed = dat[1].header['SHARPNSS'] <  0.01 #empirically derived sharpness value for good data
        except IOError:
            os.remove(fname)
            fhandle = open(fname,'wb')    
            try: #prevents file permission error from killing loop
                self.ftp.retrbinary('RETR {0}'.format(fname),fhandle.write)
            except:
                fhandle.close()
                return True
            fhandle.close()
            dat = fits.open(fname)
            failed = dat[1].header['SHARPNSS'] <  0.01 #empirically derived sharpness value for good data
            
        return failed
    



def main(start,end,nproc=4,typ='fits',verbose=False,cad=120,
         larc='/Volumes/Pegasus/jprchlik/projects/ha_filaments/gong'):
    """Main loop of program. It is written to send information to grab_gong class.
       The program requires the start and end times be python datetime objects. 
       The optional inputs are as follows: nproc is the number of processors to use while downloading (default = 4, deprecated),
       typ is the type of file to download from the ftp server (default = 'fits'),
       verbose is present but not well implemented,
       cad is the cadence to download (default = 120 minutes, so download only 1 file per 120 minutes),
       larc is the local archive directory location (default = /Volumes/Pegasus/jprchlik/projects/ha_filaments/gong)""" 
       

#make sure archive dir has an ending /
    if larc[-1] != '/': larc = larc+'/'

#try to make archive directory then cd into that directory
    try:
        arcdir = '{0}{1}'.format(larc,typdic[typ])
        os.mkdir(arcdir)
    except:
        print 'Folder Already exists'

    try:
        os.chdir(arcdir)
    except:
        if verbose:
            print 'Folder not created in proper location. Check larc variable'
        return 1

#connect to gong ftp archive
    ftp = ftplib.FTP('gong2.nso.edu','anonymous')
#change directory into archive containing gong images (default = large images)
    ftpdir = 'HA/{0}'.format(typdic[typ])
    ftp.cwd(ftpdir)
    try:
        test = grab_gong(start,end,ftp,arcdir,nproc,cad,verbose)
#close ftp when finished
        ftp.close()
#return variable class which includes the all important file list
        return test

    except:
        print 'Failed unexpectedly, closing ftp access', sys.exc_info()[0]
        ftp.close()
        raise

if __name__=="__main__":
    main(start,end)




