## imports
# external packages
import numpy as np
import pandas as pd
import scipy
# from scipy.interpolate import interp1d, interp2d
# from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import meshio
import pyvista as pv
import re
import os
import time 
import sys
import numba

# other .py files
from mag_materials import *
# from reversal_hists import *
# from late_remag_wnumba import *
# from dipole_sources import calc_all

## extra functions 
def B_time_series(file_path):
    '''
    Read in a magnetic field time series from a file.

    Returns: absolute time (s), Bz, By, Bz (nT)
    '''
    in_dat = np.loadtxt(file_path,skiprows=1,delimiter=',')

    # convert Julian day and seconds data into absolute time
    JD = in_dat[:,4]-in_dat[0,4]
    abs_time = JD*24*3600 + in_dat[:,5]
    
    return np.stack([abs_time,in_dat[:,7],in_dat[:,8],in_dat[:,9]],axis=1)

def get_sim_time_steps(tot_time,time_step,scale=1.05,max_steps=500):
    '''
    Get the time steps at which the simulation was run based on the total time and time step
    '''

    # get total number of steps        
    t_full=time_step*scale**np.arange(max_steps)

    if t_full[-1]<tot_time:
        return get_sim_time_steps(tot_time,time_step,scale=1.05,max_steps=2*max_steps)
    
    t_full=np.cumsum(t_full)
    n_steps = np.argmax([t_full>tot_time])+2

    t_full = np.insert(t_full,0,0)
    return t_full[:n_steps]

## VolMag class
class VolMag:
    def __init__(self,basefp,imfile=False, tadj=1.05,
                  heatsolfile = os.path.join('output','heat_solutions.npy')):
        self.basefp = basefp
        self.heatsolfp = self.basefp + heatsolfile
        self.tadj = tadj
        self.imfile=imfile
        

        # get important quantities from config file
        cfg = open(os.path.join(self.basefp,'config.cfg'), "r")
        list_of_lines = cfg.readlines()
        
        for line in list_of_lines:
            l_split = line.split()
            fp_regex = re.compile('[a-zA-Z0-9/.]+')
            if len(l_split) > 0:
                if l_split[0] == 'output_folder':
                    fp_regex.findall(l_split[2])
                    self.output_fp = fp_regex.findall(l_split[2])[0]
                    print('output folder: ', self.output_fp)
                if l_split[0] == 'T_surf':
                    self.T_surf = float(re.sub(';','',l_split[2]))
                    print(f'surface temperature: {self.T_surf:.2f} K')
                if l_split[0] == 'time_step':
                    self.dt0 = float(re.sub(';','',l_split[2]))*365.25*24*3600
                    print(f'initial time step: {self.dt0:.2e} s ({self.dt0/(365.25*24*3600):.2f} years)')
                if l_split[0] == 'final_time':
                    self.ttot = float(re.sub(';','',l_split[2]))*365.25*24*3600
                    print(f'total simulation time: {self.ttot:.2e} s ({self.ttot/(365.25*24*3600):.2f} years)')

        self.t_list = get_sim_time_steps(self.ttot,self.dt0,scale=tadj,max_steps=500)

        return

    def read_sol(self,save_heat=False,load_heat=False,curie_trim=True):
        output_fp = os.path.join(self.basefp,'output')
        meshi = pv.read(os.path.join(output_fp,'solution-000.vtk'))

        # get cell volumes
        meshi = meshi.compute_cell_sizes(area=False,length=False)
        Vi = meshi.cell_data['Volume']

        # get cell center locations
        cell_centers_mesh = meshi.cell_centers()
        xi=cell_centers_mesh.points[:,0]
        yi=cell_centers_mesh.points[:,1]
        zi=cell_centers_mesh.points[:,2]

        sampled = cell_centers_mesh.sample(meshi)
        Tc = sampled['U']
        
        # trim out cells below Curie depth
        if curie_trim:
            # TODO: calculate this properly
            curie_depth = -50000.

            xi = xi[zi>curie_depth]
            yi = yi[zi>curie_depth]
            Vi = Vi[zi>curie_depth]
            Tc = Tc[zi>curie_depth]
            zi_c = zi[zi>curie_depth]

            print(f'Removed {len(zi)-len(zi_c):d} cells below Curie depth.')

        # load temperature history if we saved it previously
        if load_heat:
            T_array = np.load(os.path.join(output_fp,'T_dat.npy'))

        # otherwise, build the temperature history array
        else:
            # initialize temperature array
            T_array = np.zeros([len(xi),len(self.t_list)])
            T_array[:,0] = Tc
            
            i = 1
            while os.path.isfile(os.path.join(output_fp,f"solution-{i:03d}.vtk")):
                meshi = pv.read(os.path.join(output_fp,f"solution-{i:03d}.vtk"))

                sampled = cell_centers_mesh.sample(meshi)
                Tc = sampled['U']

                if len(Tc) != len(zi):
                    raise Exception(f'Mesh at step {i:d} does not contain the same number of points as the first mesh.')

                # trim out cells below Curie depth
                if curie_trim:
                    Tc = Tc[zi>curie_depth]
                
                T_array[:,i] = Tc

                i += 1

        # n_steps = i-1
        print(f'Number of cells: {len(xi):d}')

        # store temperature, position, and volume data
        self.T_array = T_array
        self.x = xi
        self.y = yi
        self.z = zi_c
        self.V = Vi

        # save temperature array if necessary
        if save_heat:
            np.save(os.path.join(output_fp,'T_dat.npy'),T_array)

        return
    
    def interp_T(self):
        return

    def make_dM(self, M_function,del_T=False):
        M_array,self.susc = M_function(self.T_array)

        M_array = np.maximum.accumulate(M_array[:,::-1])[:,::-1]

        # calculate fraction of magnetization set during each time step
        self.dM_array = M_array[:,1:]-M_array[:,:-1]
        self.dM_array[self.dM_array<0] = 0
        # np.insert(self.dM_array,-1,0)

        if del_T:
            self.T_array = np.nan

        return 
    
    def calc_net_M_const_B(self,B_vec):
        '''
        Calculate net magnetization from a magnetic field vector time series by averaging the
        magnetic field time series within each time bin of the cooling simulation.
        '''

        # get indices in time series array corresponding to each temperature step
        B_avg = np.ones([len(self.t_list)-1,3])*B_vec

        M = (self.dM_array)@B_avg*self.susc/(1.256637e-6) #  A/M

        self.M = M

        return M
    

    def calc_net_M_sample(self,B_time_series):
        '''
        Calculate net magnetization from a magnetic field vector time series by sampling the 
        field at each time.
        '''

        return
    
    def calc_net_M_avg_t(self,B_time_series):
        '''
        Calculate net magnetization from a magnetic field vector time series by averaging the
        magnetic field time series within each time bin of the cooling simulation.
        '''

        # make B and t arrays
        t_B_in = B_time_series[:,0]
        B_in = B_time_series[:,1:]

        # get indices in time series array corresponding to each temperature step
        B_avg = np.zeros([len(self.t_list)-1,3])

        for i in range(len(self.t_list) - 1):
            t_mask = (t_B_in >= self.t_list[i]) & (t_B_in < self.t_list[i + 1])
             
            if np.any(t_mask):
                B_avg[i,:] = np.mean(B_in[t_mask],axis=0)

            # otherwise map to the closest time step
            else:
                
                t_closest = np.argmin(np.abs(t_B_in - self.t_list[i]))
                B_avg[i,:] = B_in[t_closest] 
        
        # store magnetic field time series used to calculate magnetization
        self.B_t = B_avg

        # calculate net magnetization in each cell
        M = (self.dM_array)@B_avg*self.susc/(1.256637e-6) #  A/M

        self.M = M

        return M
    


    def calc_net_M_interp_t(self,B_time_series):
        '''
        Calculate net magnetization from a magnetic field vector time series by interpolating
        T data onto times from time series
        '''

        return
    
    def calc_net_M_max_dM(self,B_time_series):
        '''
        Calculate net magnetization from a magnetic field vector time series by ?? doing
        something fancy with interpolation with a maximum dM?
        '''

        return
    
    def trim_M(self,tol):
        '''
        Remove elements from magnetization array where the net magnetization is below a threshold tol. 
        '''
        keep_inds = (np.sqrt(np.sum(self.M**2,axis=1))>tol)

        n0 = len(self.z)

        self.M = self.M[keep_inds]
        self.x = self.x[keep_inds]
        self.y = self.y[keep_inds]
        self.z = self.z[keep_inds]
        self.V = self.V[keep_inds]
        
        print(f'Removed {n0-len(self.z):d} cells with net magnetization below {tol:.2e}.')

        return

