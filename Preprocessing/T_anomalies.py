'''
Make 
'''

import numpy as np
import matplotlib.pyplot as plt


def make_flat_disk_T(x, y, z, x0, y0, z_min, z_max, tube_radius, lava_T, baseline_value=0.):
    
    # Initialize the result array with baseline values
    if isinstance(baseline_value, np.ndarray) & (baseline_value.shape == x.shape):
        result = baseline_value
    elif isinstance(baseline_value, float):
        result = np.full_like(x, baseline_value)
        
    # Find indices of points within tube_radius
    xances = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    keep_inds = (xances <= tube_radius)*(z >= z_min)*(z <= z_max)
    result[keep_inds] = lava_T
    
    return result


def make_lava_tube_T(x, y, z, start_point, end_point, tube_radius, lava_T, baseline_value=0.):
    
    # Extract coordinates
    x1, y1, z1 = start_point
    x2, y2, z2 = end_point
    
    # Initialize the result array with baseline values
    if isinstance(baseline_value, np.ndarray) & (baseline_value.shape == x.shape):
        result = baseline_value
    elif isinstance(baseline_value, float):
        result = np.full_like(x, baseline_value)
    
    # Calculate direction vector
    direction = np.array([x2-x1, y2-y1, z2-z1])
    length = np.linalg.norm(direction)
    unit_direction = direction / length
    
    # Use a fixed number of steps based on the grid size
    steps = max(x.shape[0], x.shape[1], x.shape[2])
    
    # Generate points along the line
    t = np.linspace(0, 1, steps)
    line_points = np.array([
        x1 + unit_direction[0] * length * t,
        y1 + unit_direction[1] * length * t,
        z1 + unit_direction[2] * length * t
    ])
    
    # For each point along the line
    for point in line_points.T:
        # Find indices of points within tube_radius
        xances = np.sqrt(
            (x - point[0])**2 + (y - point[1])**2 + (z - point[2])**2)
        result[xances <= tube_radius] = lava_T
    
    return result


def make_impact_T(x,y,Dtc,pitp,Tsurf,g,vi,theta,n,K0,Cp,rhot,Dq,Tsol,Tliq,plot=0):
    '''
    Calculate post-impact shock heating using equations from Abramov et al. (2013) 
    inputs
    --------------
    y: 
    x:
    Dtr: transient crater diameter (km)

    parameters
    --------------
    vi: impact velocity (km/s)
    theta: impact angle (degrees)
    K0: adiabatic bulk modulus
    Cp: heat capacity (J k^-1 C^-1)
    rhot: target density
    '''
    Dtc=Dtc*1000
    Dtr=Dtc*1.2
    y = y*1000
    x = x*1000

    # projectile radius
    Dim = ((Dtc/1.16)*(vi*1000)**(-0.44)*g**(0.22)*np.sin(theta)**(-1/3))**(1/0.78)
    Rp = Dim/2
    
    print('projectile radius: ', Rp)

    # make r array
    r = np.sqrt((y-Rp)**2+x**2)

    # specific uncompressed target volume
    V0 = 1/rhot

    # pressure decay exponent
    k = 0.625*np.log10(vi)+1.25

    # pressure at r = Rp
    A = rhot*(vi*1000)**2/4 *np.sin(theta)

    # peak pressure
    P = A*(r/Rp)**(-k)
    P[r<Rp] = 0

    # waste heat
    dEw = 0.5*(P*V0-(2*K0*V0)/n)*(1-(P*n/K0+1)**(-1/n))+(K0*V0/(n*(1-n)))*(1-(P*n/K0+1)**(1-(1/n)))
    
    # temperature
    Tshock = dEw/Cp
    Tshock[Tshock<0] = 0

    # remove excavated material
    excinds = x**2*(Dtr+4*Rp)/(Dtc**2)-0.25*(Dtr) - Rp + y >= 0
    t2 =np.where(excinds[1:,:]!=excinds[0:-1,:])
    rollinds = [x for _, x in sorted(zip(t2[1],t2[0]))] 
    
    Tsadj = np.copy(Tshock)
    for i in [x for x,_ in sorted(zip(t2[1],t2[0]))]:
        Tsadj[:,i] = np.roll(Tshock[:,i],-rollinds[i]-1)
        Tsadj[-rollinds[i]-1:,i] = 0

    # central uplift
    Df = 0.91*Dtr**(1.125)/Dq**(0.09)
    Rcp = 0.22*(Df)/2
    cudis = 0.06*(Df/1000)**(1.1)*1000

    regul = y[(x<=Rcp)*(y<=1.25*(0.25*Dtr))]
    regulr = x[(x<=Rcp)*(y<=1.25*(0.25*Dtr))]
    
    dtemp = np.zeros(np.shape(Tshock))
    normr = np.max((regulr-Rcp)**2)
    normd = np.max(1.25*(0.25*Dtr)-regul)
    dtemp[(x<=Rcp)*(y<=(1.25*(0.25*Dtr)))] = cudis*(regulr-Rcp)**2*((1.25*(0.25*Dtr))-regul)/(normr*normd)
    hhh=y+dtemp
    Tul = pitp(y+dtemp)

    # plotting :)
    if plot:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        plt.contourf(x/1000,y/1000,Tul + Tshock,[20,45,70,95,120,200,500],cmap='YlOrRd')
        plt.colorbar();
        plt.contour(x/1000,y/1000,(x)**2*(Dtr+4*Rp)/(Dtc**2)-0.25*(Dtr) - Rp + y,[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        plt.contourf(x/1000,y/1000,Tul + Tsadj,np.linspace(0,200,40),cmap='YlOrRd')
        plt.colorbar();
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
    # make temperature dictionary
    Tout = {};
    
    Tout['Tul'] = Tul 
    Tout['Tsh'] = Tshock 
    Tout['Tshadj'] = Tsadj 
    Tout['Ttot'] = Tul + Tsadj 

    Tthresh = 0.6*Tsol + 0.4*Tliq 
    Tthresh = np.maximum(Tthresh,pitp(y,Tsurf=Tsurf))
    
    Tout['Tthresh'] = Tthresh
    
    Tout['Tconvadj'] = np.minimum(Tul + Tsadj, Tthresh)
    
    return Tout