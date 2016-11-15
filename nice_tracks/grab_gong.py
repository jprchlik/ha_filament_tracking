import ftplib
import calendar
import os

typdic = {}
typdic['large'] = 'hag'


class grab_gong:

    def __init__(self,start,end,ftp,arcdir,nproc):
        #get a list of directories
        self.dlev1 = ftp.nlst()
        self.ftp = ftp
        self.end = end
        self.arcdir = arcdir
        self.nproc = nproc
# create local archive
        self.mkloc_arc(self)

    def mkloc_arc(self):
#make local subdirectories
        for i in self.dlev1: 
            mkdirs(self,i)
#create the day subdirectories
            days = calendar.monthrange(int(i[:4]),int(i[4:]))
            for j in range(1,days)+1: 
                ddir = '{0}/{0}{1:2d}'.format(i,j).replace(' ','0')
                mkdirs(self,ddir)

#Actually make the directories                 
    def mkdirs(self,adir):
        try:
            os.mkdir(adir)
        except:
            print 'Directory {0} already exists'.format(adir)
        

    def getfiles(self):
        print 'test'

def main(start,end,nproc=4,typ='large',
         larc='/Volumes/Pegasus/jprchlik/projects/ha_filaments/gong/'):

#try to make archive directory then cd into that directory
    try:
        arcdir = '{0}/{1}'.format(larc,typdic[typ])
        os.mkdir(arcdir)
    except:
        print 'Folder Already exists'

    try:
        os.cwd(arcdir)
    except:
        print 'Folder not created in proper location. Check larc variable'
        return 1

#connect to gong ftp archive
    ftp = ftplib.FTP('gong2.nso.edu','anonymous')
#change directory into archive containing gong images (default = large images)
    ftp.cwd('HA/{0}'.format(typdic[typ]))
    test = grab_gong(start,end,ftp,arcdir,nproc)

#close ftp when finished
    ftp.close()


if __name__=="__main__":
    main(start,end)




