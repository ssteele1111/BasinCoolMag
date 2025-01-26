'''
Created on Dec 1, 2021

@author: sarahcate98
'''

## imports
# external packages
import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
import scipy
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
import meshio
import time
import os

# other .py files
from mag_materials import *
from reversal_hists import *
from late_remag_wnumba import *
from dipole_sources import calc_all, B_cube_dict_choclo, cube_jacobian#, calc_point_source_field
from interps import interp_T_t
# from bgfields import BGLil10

## BasinMag class
class MakeMag:
    '''
    Class for creating magnetization arrays from cooling data.
    '''
    def __init__(self,basefp,late_remag='none', tadj=1.05,interpt=1e4,bg_mag=False,imfile=False,**kwargs):
        self.basefp = basefp
        self.makefps(**kwargs)
        self.tadj = tadj
        self.interpt = interpt
        self.bg_mag = bg_mag
        self.lr_scaling = late_remag
        self.imfile = imfile
        
        # get lines from config file
        cfg = open(self.basefp+'config.cfg', "r")
        list_of_lines = cfg.readlines()
        
        # set initial time step (dt0) and total run time (ttot)
        try:
            # set initial time step (dt0) and total run time (ttot)
            self.dt0 = float(list_of_lines[98][19:-2])
            self.ttot = float(list_of_lines[99][19:-2])
            
            # set # of prerefine steps
            self.preref = int(list_of_lines[74][27:-18])
        except:
            try:
                self.dt0 = float(list_of_lines[100][19:-2])
                self.ttot = float(list_of_lines[101][19:-2])
                
                # set # of prerefine steps
                self.preref = int(list_of_lines[75][27:-18])
            except:
                self.dt0 = float(list_of_lines[101][19:-2])
                self.ttot = float(list_of_lines[102][19:-2])
                
                # set # of prerefine steps
                self.preref = int(list_of_lines[76][27:-18])
        
        # # load in base mesh, collect mesh points into a np array
        # mesh_df = pd.read_csv(basefp + 'mesh.inp', sep=" ",names=[1,2,3,4,5,6,7,8,9,10,11], skiprows=1, engine='python');
        
        # mesh_points = mesh_df[mesh_df[3]!='quad'].astype('float')
        # self.mesh_points = mesh_points.loc[:,[3,4]].to_numpy()
        
        return
    
    def makefps(self,meshparamsfile = 'mesh_params.txt', heatsolfile = 'output/heat_solutions.txt'):
        ''' make important file paths '''
        self.meshparamsfp = self.basefp + meshparamsfile
        self.heatsolfp = self.basefp + heatsolfile
        
        return
    
    def get_dt(self):
        ''' set time step and total model run time using fractional adjsutment at each step and # of solution files '''
        
        self.dtlist = np.insert(self.tadj**np.arange(0,self.nt-1)*self.dt0,0,0)
        self.t_list = np.cumsum(self.dtlist)
        
        return
    
    def get_dt_const(self):
        ''' set time step and total model run time using config file, fractional adjsutment at each step, and # of solution files '''
        # get lines from config file
        cfg = open(self.basefp+'config.cfg', "r")
        list_of_lines = cfg.readlines()
        
        # set time step (dt) and total run time (ttot)
        self.dt = float(list_of_lines[98][19:-2])
        self.ttot = float(list_of_lines[99][19:-2])
        
        return
    
    def read_sol(self,output_fp,dx=2000,dy=False,save_heat=False,load_heat=False,curietrim=True,tcoarsen=1,xcoarsen=1):
        # store tcoarsen choice
        self.tcoarsen = tcoarsen
        
        if not dy:
            dy = dx
            
        # load initial mesh to extract dimensions, etc.
        meshi = meshio.read(output_fp+'/solution-000.vtk')
            
        # get x, y, T data
        xi=meshi.points[:,0]
        yi=meshi.points[:,1]
        Ti = meshi.point_data['U']
        
        new_x = np.arange(np.min(xi),np.max(xi),dx)
        new_y = np.arange(np.max(yi),np.min(yi),-dy)
        
        t_full=100*1.05**np.arange(500)
        t_full=np.cumsum(t_full)
        n_steps = np.argmax([t_full>self.ttot])+2
        
        if load_heat:
            # n_steps = len(os.listdir(output_fp))-1
            heat_sols = np.load(output_fp+'/T_reg.npy')
            heat_sols = heat_sols.reshape(heat_sols.shape[0],-1,n_steps)
            
                # print("Heat file not found. Generating anew")
                # self.read_sol(output_fp,dx=dx,save_heat=True,load_heat=False,curietrim=curietrim,tcoarsen=tcoarsen,xcoarsen=xcoarsen)
                # return self.heat_sols
            
        else:
            heat_sols = np.zeros((len(new_y),len(new_x),n_steps))
            
            new_x_mesh,new_y_mesh = np.meshgrid(new_x,new_y)
            
            xy_out = np.stack([new_x_mesh.flatten(),new_y_mesh.flatten()]).T
            
            i =  0
            while os.path.isfile(output_fp + f'\solution-{i:03d}.vtk'):
                # load mesh
                meshi = meshio.read(output_fp + f'\solution-{i:03d}.vtk')
                
                print(i)
                # get x, y, T data
                xi=meshi.points[:,0]
                yi=meshi.points[:,1]
                xy_in = np.stack([xi,yi]).T
                try:
                    Ti=meshi.point_data['solution']
                except:
                    Ti=meshi.point_data['U']
                    
                # interpolate onto set mesh
                T_out = scipy.interpolate.griddata(xy_in, Ti, xy_out)
                
                # add to bigger array
                heat_sols[:,:,i] = T_out.reshape((len(new_y),len(new_x)))
                
                
                i += 1
            
        if save_heat:
            np.save(output_fp+'/T_reg.npy',heat_sols.reshape(heat_sols.shape[0], -1))
            print('done!)')
        
        # reorient heat_sols for rest of script
        heat_sols = np.moveaxis(heat_sols,2,0)
        
        
        # trim and subsample array as necessary
        if curietrim and np.any(heat_sols[0,:,-1]>1200):
            cutind = np.where(heat_sols[0,:,-1]>1200)[0][0]
            self.xvals = new_x[::xcoarsen]
            self.zvals = new_y[:cutind:xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = n_steps
            self.heat_sols = heat_sols[::tcoarsen,:cutind:xcoarsen,::xcoarsen]
            
        else:   
            self.xvals = new_x[::xcoarsen]
            self.zvals = new_y[::xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = n_steps
            self.heat_sols = heat_sols[::tcoarsen,::xcoarsen,::xcoarsen]
        
                           
        return self.heat_sols

    def read_sol_txt(self,rdepth=1,cdepth=50,ldepth=100,save_heat=False,load_heat=False, curietrim=True,tcoarsen=1,xcoarsen=1):
        '''
        read solution files and make heat solutions array.
        
        curietrim: cut off section of matrix that stays above ~hematite Curie point at all time steps
        tcoarsen: only keep every n time stdeps
        xcoarsen: only keep every n grid points
        '''
        # store tcoarsen choice
        self.tcoarsen = tcoarsen
        
        # load heat solutions file
        heat_sols_raw = np.loadtxt(self.heatsolfp)
        
        # drop prerefine rows
        heat_sols_raw = np.concatenate([heat_sols_raw[0].reshape(1,-1),heat_sols_raw[self.preref*2+1:]])
        
        print('Done 1')
        
        # load mesh parameters
        mps = np.loadtxt(self.meshparamsfp)
        
        print('Done 2')
        xvals = mps[:,0]
        zvals = mps[:,1]
        xv_un = np.unique(xvals)
        zv_un = np.unique(zvals)
        
        # reshape heat solutions
        dimx = xv_un.shape[0]
        dimz = zv_un.shape[0]
        heat_sols_rs = np.reshape(heat_sols_raw,(-1,dimz,dimx))

        # initialize final heat solutions array
        heat_sols = np.zeros(heat_sols_rs.shape)
        
        # deal with weird solutions things resulting from different material properties
        if np.max(np.abs(zv_un)) <= rdepth*1000:
            ind_dr = np.max(np.where(zv_un > -rdepth*1000)[0]) - np.min(np.where(zv_un > -rdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            
            for i in range(heat_sols_rs.shape[0]):
                temp = np.concatenate((heat_sols_rs[i,ind_dr+2:],np.roll(heat_sols_rs[i],-ind_dr-2,axis=0)[ind_r-1:ind_r+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dr+2]))
                
                temp = temp[::-1,:]
                
                heat_sols[i,:,:] = temp
                
        elif np.max(np.abs(zv_un)) <= cdepth*1000:
            ind_dc = np.max(np.where(zv_un > -cdepth*1000)[0]) - np.min(np.where(zv_un > -cdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0]);
            
            for i in range(heat_sols_rs.shape[0]):
                temp = np.concatenate((heat_sols_rs[i,ind_dc+2:],np.roll(heat_sols_rs[i],-ind_dc-2,axis=0)[ind_c-1:ind_c+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dc+2]))
                
                
                temp = temp[::-1,:]
                
                heat_sols[i,:,:] = temp
                
                
        elif np.max(np.abs(zv_un)) <= ldepth*1000:
            ind_dc = np.max(np.where(zv_un > -cdepth*1000)[0]) - np.min(np.where(zv_un > -cdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0]);
            
            for i in range(heat_sols_rs.shape[0]):
                hs_temp = heat_sols_rs[i]
                
                hs_temp[:2,:]=hs_temp[:2,:].flatten().reshape(2,-1,order='F')
                hs_temp[2:4,:]=hs_temp[2:4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:]=hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:].flatten().reshape(2,-1,order='F')
                
                hs_temp = hs_temp[::-1,:]
                
                hs_temp = np.concatenate((hs_temp[-2:,:],
                               hs_temp[ind_c-1:-2,:],
                               hs_temp[:ind_c-1,:]))
                
                hs_temp[0,0],hs_temp[1,1] = hs_temp[1,1],hs_temp[0,0]
                
                heat_sols[i,:,:] = hs_temp
                
        else:
            ind_dl = np.max(np.where(zv_un > -ldepth*1000)[0]) - np.min(np.where(zv_un > -ldepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0])
            ind_l = np.min(np.where(zv_un > -ldepth*1000)[0])
            
            for i in range(heat_sols_rs.shape[0]):
                # temp = np.concatenate((heat_sols_rs[i,ind_dl+2:],np.roll(heat_sols_rs[i],-ind_dl-2,axis=0)[ind_l-1:ind_l+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dl+2]))
                
                hs_temp = heat_sols_rs[i]
                
                hs_temp[:2,:]=hs_temp[:2,:].flatten().reshape(2,-1,order='F')
                hs_temp[2:4,:]=hs_temp[2:4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:]=hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_l+2:ind_r-ind_l+4,:]=hs_temp[ind_r-ind_l+2:ind_r-ind_l+4,:].flatten().reshape(2,-1,order='F')
                
                hs_temp = hs_temp[::-1,:]
                
                hs_temp = np.concatenate((hs_temp[-2:,:],
                               hs_temp[ind_c-1:-2,:],
                               hs_temp[ind_l-1:ind_c-1,:],
                               hs_temp[:ind_l-1,:]))
                
                hs_temp[0,0],hs_temp[1,1] = hs_temp[1,1],hs_temp[0,0]
                
                heat_sols[i,:,:] = hs_temp
          

        # get properly ordered z values for final arrays
        zuinds = np.unique(zvals, return_index=True)[1]
        zv_un = np.array([zvals[ind] for ind in sorted(zuinds)])
        
        # trim and subsample array as necessary
        if curietrim and np.any(heat_sols[0,:,-1]>1200):
            cutind = np.where(heat_sols[0,:,-1]>1200)[0][0]
            self.xvals = xv_un[::xcoarsen]
            self.zvals = zv_un[:cutind:xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = heat_sols.shape[0]
            self.heat_sols = heat_sols[::tcoarsen,:cutind:xcoarsen,::xcoarsen]
            
        
        else:   
            self.xvals = xv_un[::xcoarsen]
            self.zvals = zv_un[::-xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = heat_sols.shape[0]
            print('!!!!')
            self.heat_sols = heat_sols[::tcoarsen,::xcoarsen,::xcoarsen]
        
        
        
        return self.heat_sols
    
    def make_map_arrays(self):
        ''' makes arrays for basin in full 3D '''
        # make x, y, r arrays
        
        xy = np.meshgrid(self.xvals,self.xvals)
        rmap = np.sqrt(xy[0]**2+xy[1]**2)
        
        # map elements of r array to closest value in x array
        rmap_inds = np.argmin(np.abs(rmap[:,:,None]-self.xvals[None,None,:]),axis=2);
        
        self.xy = xy
        self.rmap = rmap
        self.rmap_inds = rmap_inds
        return rmap  
     
    def make_dM(self, Mfunc, heat_in=False, tmin=0, tmax=1e10):
        ''' returns remagnetized fraction at each time step and set important magnetic properties'''
        
        if isinstance(heat_in,bool):    
            heat_in = self.heat_sols
            
        mag_props = Mfunc(heat_in)
        Mseries = mag_props[0]
        
        # set object magnetic properties
        self.Mfunc = Mfunc
        self.dM_mat = Mseries               # fractional remagnetization 
        self.susc = mag_props[1]            # susceptibility 
        self.maxdM = np.max(Mseries,axis=0) # maximum fractional remagnetization (for implementing background fields)
        self.ieq = -1   #np.searchsorted(self.tintlist, tmax)-1
        
        return Mseries
    
    # def make_Particles(self,epsilon=1e-6):
    #     '''
    #     Makes Particle objects everywhere where the maximum magnetization set during cooling is above ~0. 
    #     '''
        
    #     # initialize particle list
    #     particles = []
        
    #     # loop over domain and create particles
    #     # this is slow but we only have to do it once :)
    #     for z,k in enumerate(self.zvals):
    #         for x,i in enumerate(self.xvals):
    #             if (self.maxdM[i,k]**2 >= epsilon**2):
    #                 for y,j in enumerate(self.xvals):
    #                     particles.append(Particle(coords=(x,y,z),m=(0,0,0)))
                        
    #     print(f'Final particle count: {len(particles):d}')
        
    #     self.particles = particles
        
    #     return particles
    
    def get_cell_dxs(self,type='reg_grid'):
        if type == 'reg_grid':
            un_dx = np.abs(self.xvals[1]-self.xvals[0])
            un_dz =  un_dx
        elif type == 'cuboid':
            un_dx = np.abs(np.unique(self.xvals[::-1]- self.xvals[1::]))
            un_dz = np.abs(np.unique(self.zvals[::-1]- self.zvals[1::]))
        return un_dx, un_dz
    
    def B_calc(self,type):
        """_summary_

        Args:
            type (str): _description_ ['reg_grid', 'cuboid']
        """
        if type == 'reg_grid':
            self.B_calc_reg_grid()
        elif type == 'cuboid':
            self.B_calc_cuboid()
        return
    
    def thing_mat(self,B_hist):
        '''
        B_hist: array containing magnetic field history with each row (t,Bx,By,Bz)
        '''
        B_hist = B_hist.reshape(-1,4)

        # interpolate the cooling curve at each point onto same time series as B_hist
        self.t_int_list = B_hist[:,0]
        self.heat_sols = interp_T_t(self.heat_sols,self.T_max,self.t_list,self.t_int_list)
        
        # map to unblocking spectrum
        self.make_dM(self.Mfunc)

        delta_M = (self.dM_mat[1:,:]-self.dM_mat[:-1,:])

        # sum over the magnetic history in all three dimensions to get the net vector magnetization
        Mx = np.sum(delta_M*(B_hist[1:,1]+B_hist[:-1,1])/2,axis=0)
        My = np.sum(delta_M*(B_hist[1:,2]+B_hist[:-1,2])/2,axis=0)
        Mz = np.sum(delta_M*(B_hist[1:,3]+B_hist[:-1,3])/2,axis=0)

        # return array of magnetization vectors
        return Mx, My, Mz
        
    def make_old_mag(self):
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        # flatten r map array
        rinds = self.rmap.flatten()
        
        # create interpolations
        itrpMmax = interp2d(self.xvals,self.zvals,self.maxdM)

        # sort array, make recovery index
        sorted_r = np.sort(rinds)
        recovinds_r = np.argsort(np.argsort(rinds))
        
        # evaluate interpolation and put back in usual coordinates
        quadr_mag_frac = itrpMmax(sorted_r,self.zvals)[:,recovinds_r].reshape(lz,lx,lx)
        
        self.olddM_quadr = 1-quadr_mag_frac
        
        mag_moms = quadr_mag_frac[::-1,:,:]
        mag_moms = np.concatenate((mag_moms[:,:0:-1,:],mag_moms),axis=1)
        mag_moms = np.concatenate((mag_moms[:,:,:0:-1],mag_moms),axis=2)
        return mag_moms
    
    
    def make_M_map(self,dM_arr):
        '''
        Makes map of net magnetization magnitude in one quadrant
        
        dM_arr: output of self.Mnet(revR)
        '''
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        # flatten r map array
        rinds = self.rmap.flatten()
        
        # create interpolations
        itrpM = interp2d(self.xvals,self.zvals,dM_arr)

        # sort array, make recovery index
        sorted_r = np.sort(rinds)
        recovinds_r = np.argsort(np.argsort(rinds))
        
        # evaluate interpolation and put back in usual coordinates
        quadr_mag_frac = itrpM(sorted_r,self.zvals)[:,recovinds_r].reshape(lz,lx,lx)
    
        return quadr_mag_frac
    
    
    def late_remag(self):
        if self.lr_scaling == 'none':
            self.lr_out = np.zeros((len(self.zvals),len(self.xvals)*2-1,len(self.xvals)*2-1))
            
            return 
        
        elif self.lr_scaling == 'load':
            
            # fp_lr = r'C:\Users\SteeleSarah\Researches\ImpactCooling\ModelRuns\Tissint\Cold\ForFigs\BmapsFig\impacts_MBALate_1_1000km_010.npy'
            fp_lr = rf'C:\Users\SteeleSarah\Researches\ImpactCooling\Scripts\ImpactMaps\1e6km_2x2km\smalls_{np.random.randint(32):d}.npy'
            print('Loading late basins from: ' + fp_lr)
            lr_d_mag = np.load(fp_lr)
            
            self.lr_out = lr_d_mag.transpose(2, 0, 1)
            return
        
        elif self.lr_scaling == 'excavation':  
            print('Adding late impacts...')
            xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
            xyz = np.meshgrid(xv_whole,xv_whole,self.zvals)
            
            # for ii in range(10):
            lr_D = ImDomain(xyz[0],xyz[1],xyz[2])
            lr_D.run_all(4.1,MBA_late,MPF_MBA,scaling=self.lr_scaling)
            self.lr_out = lr_D.mag.transpose(2, 0, 1)
            self.lr_out = self.lr_out[:,::-1,::-1]
                
            
            return
    
    def do_setup(self,Mfunc,output_fp,**kwargs):
        ''' do essential setup for basin '''
        # read heat solutions output file
        try:
            self.read_sol_txt(**kwargs)
        except:
            self.read_sol(output_fp,**kwargs)
        
        # make time list
        self.get_dt()
        
        # interpolate to new time grid
        self.t_int_list = np.arange(0,self.ttot,self.interpt)
        self.heat_sols = interp_T_t(self.heat_sols,self.T_max,self.t_list,self.t_int_list)
        
        print(self.heat_sols.shape)
        
        
        # make arrays for magnetic mapping
        self.make_map_arrays()
        
        # calculate remagnetized fraction for each pixel at each time step
        self.make_dM(Mfunc)
        
        
        # calculate fraction of magnetization that is old
        self.make_old_mag()
        
        # write params file
        self.write_params()
        
        return
    
    def write_params(self):
        ''' write mapping parameters file '''
        # set filepath
        params_fp = self.basefp + 'params.txt'
        
        # write file
        with open(params_fp, 'w') as f:
            f.write('dim_x: %d \n' % len(self.xvals))
            f.write('dim_z: %d \n' % len(self.zvals))
            f.write('dt0: %1.6e \n' % self.dt0) 
            f.write('t_tot: %1.6e \n' % self.ttot) 
            
        return
    