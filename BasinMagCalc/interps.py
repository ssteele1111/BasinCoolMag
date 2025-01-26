import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def interp_T_t(heat_sols,T_max,t_list,t_int_list):
    # smooth out oscillation artifacts from modeling
    heat_sols_temp = savgol_filter(heat_sols,2,1, axis=0)
    
    # fix undershoot
    heat_sols_temp[heat_sols_temp<T_max] = T_max
    
    # interpolate onto regular time grid
    T_t_int = interp1d(t_list,heat_sols_temp,axis=0)
        
    heat_sols_int = T_t_int(t_int_list)

    return heat_sols_int
