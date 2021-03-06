import ftplib
import sys
import calendar
from datetime import datetime,timedelta
import os
from multiprocessing import Pool

typdic = {}
typdic['large'] = 'hag'
typdic['fits'] = 'haf'


class grab_gong:

    def __init__(self,start,end,ftp,arcdir,nproc,skip,verbose):
        #get a list of directories
        self.dlev1 = ftp.nlst()
        self.ftp = ftp
        self.ftpdir = ftp.pwd()
        self.end = end
        self.start = start
        self.arcdir = arcdir
        self.nproc = nproc
        self.verbose = verbose
        self.skip = skip
# create local archive
        self.mkloc_arc()
#grab the ftp archive files in range
        self.getfiles()

    def mkloc_arc(self):
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
        try:
            os.mkdir(adir)
        except:
            if self.verbose:
                print 'Directory {0} already exists'.format(adir)
        

#get files in date range from ftp archive
    def getfiles(self):
    # list of days in range as datetime object
        dates = [ self.start +timedelta(n) for n in range(int ((self.end - self.start).days))]
    #create day file list
        self.filelist = []
    #get all files in date range
        for i in dates:
            fulldir = self.ftpdir+'/'+i.strftime('%Y%m/%Y%m%d')
            os.chdir(self.arcdir+'/'+i.strftime('%Y%m/%Y%m%d'))
            self.ftp.cwd(fulldir)
            templist = self.ftp.nlst() #get list of files on server
            templist = [s for s in templist if "L" not in s] #skip Learmonth, Australia because they are not rotated properly
            templist = templist[::self.skip]#grab only every so many returned files
            for j in templist:
                self.write_loc_files(j)
                self.filelist.append(self.arcdir+'/'+i.strftime('%Y%m/%Y%m%d')+'/'+j)
 
    #move back to parent directory
    #retrieve files from server
#            pool = Pool(processes=self.nproc)
#            out = pool.map(self.write_loc_files,templist)
#            pool.close()
        
        
        
    def write_loc_files(self,fname):
   
#check to see if file exists 
        testfile = os.path.isfile(fname)
        

#if file does not exist 
        if testfile == False:
            fhandle = open(fname,'wb')    
            self.ftp.retrbinary('RETR {0}'.format(fname),fhandle.write)
            fhandle.close()



def main(start,end,nproc=4,typ='fits',verbose=False,skip=100,
         larc='/Volumes/Pegasus/jprchlik/projects/ha_filaments/gong'):

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
        print 'Folder not created in proper location. Check larc variable'
        return 1

#connect to gong ftp archive
    ftp = ftplib.FTP('gong2.nso.edu','anonymous')
#change directory into archive containing gong images (default = large images)
    ftpdir = 'HA/{0}'.format(typdic[typ])
    ftp.cwd(ftpdir)
    try:
        test = grab_gong(start,end,ftp,arcdir,nproc,skip,verbose)
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




