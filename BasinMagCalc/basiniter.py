'''
Sarah Steele
2023
'''

# imports
import matplotlib.pyplot as plt
import time
import os
from numpy.random import randint

from basinremag import *
from mag_materials import *
from reversal_hists import Bprofile


# preset image grid
imgrid2=np.meshgrid(np.arange(-40e3,41e3,10e3),np.arange(-750e3,751e3,20e3))
imgrid3=np.meshgrid(np.arange(-40e3,41e3,10e3),np.arange(-1250e3,1251e3,20e3))
imgrid5=np.meshgrid(np.arange(-750e3,751e3,25e3),np.arange(-750e3,751e3,25e3))
imgrid6=np.meshgrid(np.arange(-500e3,501e3,15e3),np.arange(-500e3,501e3,15e3))
imgrid9=np.meshgrid(np.arange(-1250e3,1251e3,50e3),np.arange(-1250e3,1251e3,50e3))
imgridZ=[np.outer(np.cos(np.linspace(0,2*np.pi+0.1,100)),np.array([.7,.75,.8,.85]))*1100e3,
np.outer(np.sin(np.linspace(0,2*np.pi+0.1,100)),np.array([.7,.75,.8,.85]))*1100e3]

# set input/output files

fpout = os.path.join(os.getcwd(),'mag_output')
fpin = os.path.join(os.getcwd(),'basins')

# make output file if it doesn't exist
if not os.path.isdir(fpout):
        os.mkdir(fpout)

# main function
def do_basiniter_dual(basinsizes,nRevs,mu,thresh,magdir,magdirstr,liftoff,imgrid,soldx=1000,tcoarsen=1,xcoarsen=1,curietrim=True,load_heat=False,save_heat=False,fileapp='',late_remag='none',bg_mag=False,imfile=False):
    '''
    Iterates over specified basin sizes and saves magnetic field maps, susceptibility map slices, and reversal histories for
    a given number of randomly generated reversal histories shared by all basin sizes.
    
    basinsizes: list of basin sizes, must match file names
    nRevs: number of reversal histories to generate
    mu, thresh: parameters for relevant binomial distribution
    magdir: magnetization direction in normalized [x,y,z] format
    magdirstr: save file string
    liftoff: height above surface to perform mapping
    imgrid: grid of points over which to calculate magnetic fields
    '''
    # make base dir if necessary
    if not os.path.isdir(os.path.join(fpout,'BMaps')):
        os.mkdir(os.path.join(fpout, 'BMaps'))
        os.mkdir(os.path.join(fpout,'BMaps_nolr'))
        os.mkdir(os.path.join(fpout, 'BMaps_no2k'))
        os.mkdir(os.path.join(fpout,'RevRates'))
        os.mkdir(os.path.join(fpout,'Revs'))
            
    # make reversal folder path if necessary
    if not os.path.isdir(fpout + 'Revs' + os.sep+magdirstr):
        os.mkdir(fpout + 'Revs'+os.sep+ magdirstr)
        
    # make reversal folder path if necessary
    if not os.path.isdir(os.path.join(fpout,'Revs', magdirstr,'Full')):
        os.mkdir(os.path.join(fpout,'Revs', magdirstr,'Full'))
               
    # make common reversal histories
    RevsC = []
    
    for j in range(nRevs):
        RevsC.append(B_revs(mu, thresh, Nt=60000))
        
            
    for i in basinsizes:
        t0 = time.time()
        
        input_fp = os.path.join(fpin,rf'{i:d}km')
        # make basin
        bmt1 = BasinMag(input_fp,late_remag=late_remag)
        
        print('Starting ' + str(i) + 'km')
        RevsB = []
        
        # make label
        outtag = str(i)+'km_' + str(mu) +'_'+str(thresh) + '_' + str(nRevs) 
        output_fp = os.path.join(input_fp,'output')
        bmt1.do_setup(M_Tissint,output_fp,dx=soldx,tcoarsen=tcoarsen,xcoarsen=xcoarsen,curietrim=curietrim,load_heat=load_heat,save_heat=save_heat)
        
        # get reversals on basin timescale
        nmax = bmt1.ttot/1e8*10000 - 1 # find highest location in full reversal history array that overlaps cooling history
        nadj = 1e4/(bmt1.interpt*1)        # scale reversal history to map to time steps used in cooling model + adjust for coarsening
        
        # initialize reversal rates array
        revRates = np.zeros(len(RevsC))
        
        for k,revk in enumerate(RevsC):
            maxk = RevsC[k]
            scaledk = maxk[maxk<nmax]*nadj
            
            revsk=np.unique(scaledk.astype(int))
            
            RevsB.append(revsk)
            print(len(revsk)/bmt1.ttot)
            
            # calculate reversal rate
            revRates[k] = len(revsk)/bmt1.ttot
            
        
        # make dir if necessary
        if not os.path.isdir(os.path.join(fpout,'BMaps',magdirstr)):
            os.mkdir(os.path.join(fpout,'BMaps',magdirstr))
        if not os.path.isdir(os.path.join(fpout + 'BMaps_nolr',magdirstr)):
            os.mkdir(os.path.join(fpout,'BMaps_nolr',magdirstr))
        if not os.path.isdir(os.path.join(fpout + 'BMaps_no2k',magdirstr)):
            os.mkdir(os.path.join(fpout,'BMaps_no2k',magdirstr))
        if not os.path.isdir(os.path.join(fpout + 'RevRates',magdirstr):
            os.mkdir(os.path.join(fpout,'RevRates',magdirstr))
        
        # save full reversal histories
        fp_R0 = os.path.join(fpout,'Revs',magdirstr,'Full','Revs_' + outtag + '.txt')
        np.save(fp_R0, RevsB)
        
        if isinstance(liftoff, list):
            
            fpRR = os.path.join(fpout,'RevRates',magdirstr,'RR_' + outtag + '.txt')
            
            for i in range(len(liftoff)):
                
                # make filepaths
                fpB_0 = os.path.join(fpout,'BMaps_nolr',magdirstr,'BMap_'+outtag+'.txt')
                fpBz_0 = os.path.join(fpout,'BMaps_nolr',magdirstr,'BzMap_'+outtag+'.txt')
                fpBns = os.path.join(fpout,'BMaps_no2k',magdirstr,'BMap_'+outtag+'.txt')
                fpBzns = os.path.join(fpout,'BMaps_no2k',magdirstr,'BzMap_'+outtag+'.txt')
                fpB = os.path.join(fpout,'BMaps',magdirstr,'BMap_'+outtag+'.txt')
                fpBz = os.path.join(fpout,'BMaps',magdirstr,'BzMap_'+outtag+'.txt')
                fpRR = os.path.join(fpout,'RevRates',magdirstr,'RR_'+outtag + '.txt')
                 
                # run stuff
                revs1,B1_0,Bz1_0,B1,Bz1,Bns1,Bzns1=bmt1.do_mult_revs_dual(RevsB,magdir,0,liftoff[i],imgrid)
                
                # save stuff
                np.savetxt(fpB_0,B1_0.reshape(B1_0.shape[0],-1)) 
                np.savetxt(fpBz_0,Bz1_0.reshape(Bz1_0.shape[0],-1)) 
                np.savetxt(fpB,B1.reshape(B1.shape[0],-1)) 
                np.savetxt(fpBz,Bz1.reshape(Bz1.shape[0],-1)) 
                np.savetxt(fpBns,Bns1.reshape(Bns1.shape[0],-1)) 
                np.savetxt(fpBzns,Bzns1.reshape(Bzns1.shape[0],-1)) 
                np.savetxt(fpRR,revRates)
                
                
        else:
            revs1,B1_0,Bz1_0,B1,Bz1=bmt1.do_mult_revs_dual(RevsB,magdir,0,liftoff,imgrid)
            
            # make filepaths
            fpB_0 = os.path.join(fpout,'BMaps_nolr',magdirstr,'BMap_'+outtag+'.txt')
            fpBz_0 = os.path.join(fpout,'BMaps_nolr',magdirstr,'BzMap_'+outtag+'.txt')
            fpBns = os.path.join(fpout,'BMaps_no2k',magdirstr,'BMap_'+outtag+'.txt')
            fpBzns = os.path.join(fpout,'BMaps_no2k',magdirstr,'BzMap_'+outtag+'.txt')
            fpB = os.path.join(fpout,'BMaps',magdirstr,'BMap_'+outtag+'.txt')
            fpBz = os.path.join(fpout,'BMaps',magdirstr,'BzMap_'+outtag+'.txt')
            fpRR = os.path.join(fpout,'RevRates',magdirstr,'RR_'+outtag + '.txt')
            
            # save stuff
            np.savetxt(fpB_0,B1_0.reshape(B1_0.shape[0],-1)) 
            np.savetxt(fpBz_0,Bz1_0.reshape(Bz1_0.shape[0],-1)) 
            np.savetxt(fpB,B1.reshape(B1.shape[0],-1)) 
            np.savetxt(fpBz,Bz1.reshape(Bz1.shape[0],-1)) 
            np.savetxt(fpBns,Bns1.reshape(Bns1.shape[0],-1)) 
            np.savetxt(fpBzns,Bzns1.reshape(Bzns1.shape[0],-1)) 
            np.savetxt(fpRR,revRates)  
       
        print('Done!')
        print(str(i) + 'km time: ' + str(time.time()-t0))
        
        
    return 

# run things
do_basiniter_dual([800], 1,10,50,[0, 1, 0],"010_200km_ig3",200,imgrid3,xcoarsen=2,soldx=1000,load_heat=True,fileapp="",late_remag="excavation")
