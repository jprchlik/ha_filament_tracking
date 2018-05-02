import urllib
from datetime import datetime,timedelta
import os
import itertools
from multiprocessing import Pool






class download:

    def __init__(self,stime,etime,caden,b_dir,syn_arch='http://jsoc.stanford.edu/data/aia/synoptic/nrt/',d_wav=[193],
                 w_fmt='{0:04d}.fits',nproc=8,f_dir='{0:%Y/%m/%d/H%H00/AIA%Y%m%d_%H%M%S_}'):
    
        """
    
        Parameters:
        -----------
        stime: datetime object
            The time to start downloading AIA files.
      
        etime: datetime object
            The time to end downloading AIA files.
        
        caden: datetime time delta object 
            The time cadence to download AIA files.
        
        b_dir: string
            The base directory for locally storing the AIA archive.
        
        syn_arch: string, optional
            Location of online syntopic archive (default = 'http://jsoc.stanford.edu/data/aia/synoptic/nrt/').
        
        d_wav: list, optional
            List of wavelengths to download from the online archive (Default = [193]).
        
        w_fmt: list, optional
            The wavelength format of the wavelength list (defatult = '{0:04d}.fits').
        
        f_dir : string, optional
            Create local directory path format for SDO/AIA files (default = '{0:%Y/%m/%d/H%H00/AIA%Y%m%d_%H%M_}')
        """
    
        self.stime = stime
        self.etime = etime
        self.caden = caden
        self.b_dir = b_dir
        self.syn_arch = syn_arch
        self.d_wav = d_wav
        self.w_fmt = w_fmt
        self.nproc = nproc
        self.f_dir = f_dir
    
        
        #desired cadence for the observations
        real_cad = [result for result in self.des_cad(stime,etime,caden)]
        
        #create a list of combination of dates and wavelengths
        inpt_itr = list(itertools.product(real_cad,d_wav))

        #add additional variables to intp_itr to allow for par processing
        par_list = []
        for i in inpt_itr:
            par_list.append(i+(w_fmt,f_dir,b_dir,syn_arch))
        
        #Download the files locally in parallel if nproc greater than 1
        if self.nproc < 2:
            for i in par_list: wrap_download_file(i)
        else:
            pool = Pool(processes=self.nproc)
            outp = pool.map(wrap_download_file,par_list)
            pool.close()
            pool.join()

    #retrieve desired cadence from file list
    def des_cad(self,start,end,delta):
        """Create an array from start to end with desired cadence"""
        curr = start
        while curr < end:
            yield curr
            curr += delta

#wrapper for download file for par. processing
def wrap_download_file(args):
    return download_file(*args)

#download files from archive for each wavelength
def download_file(time,wavl,w_fmt,f_dir,b_dir,syn_arch):

   #format wavelength
   w_fil = w_fmt.format(wavl)
   #format input time
   s_dir = f_dir.format(time)

   #local output directory
   o_dir =b_dir+'/'.join(s_dir.split('/')[:-1])

   #check if direcory exists
   if not os.path.exists(o_dir):
       os.makedirs(o_dir)
  
   #create output file
   o_fil = b_dir+s_dir.split('/')[-1]+w_fil
   o_fil = b_dir+s_dir+w_fil
   #file to download from archive
   d_fil = syn_arch+s_dir+w_fil

   #remove file if program downloaded empty file 2018/04/23 J. Prchlik
   if os.path.isfile(o_fil):
       if os.path.getsize(o_fil) < 400 : os.remove(o_fil)

   #check if output file exists
   if not os.path.isfile(o_fil):

       #try to download file if fails continue on
       try:
           urllib.urlretrieve(d_fil,o_fil) 
       except:
           print("Cound not Download {0} from archive".format(d_fil))
       
       #file if program downloaded empty file 2018/04/23 J. Prchlik
       if os.path.getsize(o_fil) < 400 : os.remove(o_fil)

