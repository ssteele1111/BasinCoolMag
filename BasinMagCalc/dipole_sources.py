import numpy as np
import numba
import choclo
from math import sqrt as msqrt
from math import atan2
# from joblib import Parallel, delayed
import magpylib as magpy
from scipy.constants import mu_0 as MU0
from math import pi as PI


def calc_all(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        location: np.ndarray,
        source_vector: np.ndarray,
        lift_off: float = 0.0,
        calc_type: str = 'magpy') -> np.ndarray:
    
    if calc_type == 'dipole_sum':
        return calc_all_dipoles(x_grid,y_grid,location,source_vector,lift_off=lift_off)
    
    elif calc_type == 'magpy':
        return calc_all_magpy(x_grid,y_grid,location,source_vector,lift_off=lift_off)
    

def calc_all_magpy(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        location: np.ndarray,
        source_vector: np.ndarray,
        lift_off: float = 0.0) -> np.ndarray:
    
    z_grid = np.ones(x_grid.shape)*lift_off
    dx = np.abs(location[0,0,1]-location[0,0,0])
    
    # build observer array
    obs_arr = np.stack([x_grid.flatten(),y_grid.flatten(),z_grid.flatten()])
    
    B = magpy.getB(
        sources="Cuboid",
        observers=obs_arr,
        dimension=(dx,dx,dx),
        position=location,
        polarization=source_vector)
    
    return B


# def calc_all_dipoles(
#         x_grid: np.ndarray,
#         y_grid: np.ndarray,
#         location: np.ndarray,
#         source_vector: np.ndarray,
#         lift_off: float = 0.0) -> np.ndarray:

#     n3 = int(len(location)/10)
#     #locs = [location[:n2,:],location[n2:,:]]
#     #scs = [source_vector[:n2,:],source_vector[n2:,:]]
#     results = Parallel(n_jobs=10)([delayed(calc_point_source_field)(x_grid,y_grid,location[:n3,:],source_vector[:n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[n3:2*n3,:],source_vector[n3:2*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[2*n3:3*n3,:],source_vector[2*n3:3*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[3*n3:4*n3,:],source_vector[3*n3:4*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[4*n3:5*n3,:],source_vector[4*n3:5*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[5*n3:6*n3,:],source_vector[5*n3:6*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[6*n3:7*n3,:],source_vector[6*n3:7*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[7*n3:8*n3,:],source_vector[7*n3:8*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[8*n3:9*n3,:],source_vector[8*n3:9*n3,:],lift_off),
#                                    delayed(calc_point_source_field)(x_grid,y_grid,location[9*n3:,:],source_vector[9*n3:,:],lift_off)])

#     bx_tot = results[0][0] + results[1][0] + results[2][0] + results[3][0] + results[4][0] + results[5][0] + results[6][0] + results[7][0] + results[8][0] + results[9][0]
#     by_tot = results[0][1] + results[1][1] + results[2][1] + results[3][1] + results[4][1] + results[5][1] + results[6][1] + results[7][1] + results[8][1] + results[9][1]
#     bz_tot = results[0][2] + results[1][2] + results[2][2] + results[3][2] + results[4][2] + results[5][2] + results[6][2] + results[7][2] + results[8][2] + results[9][2]
    
#     return bx_tot,by_tot,bz_tot

@numba.njit(fastmath=True)
def calc_point_source_field(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        location: np.ndarray,
        source_vector: np.ndarray,
        lift_off: float = 0.0) -> np.ndarray:
    """
    Compute the field of a magnetic dipole point source

    Parameters
    ----------
    x_grid:  ndarray(pixel, pixel)
        grid to calculate the fields for
    y_grid: ndarray(pixel, pixel)
        grid to calculate the fields on
    location: ndarray (n_sources, 3)
        x,y,z-location of source
        z distance, not including the sensor height
    source_vector: ndarray(n_sources,3)
        xyz-components of vector not n sources
    lift_off: float
        distance between sensor and sample

    Note: rewritten in tensorflow, nut is not faster, 
    vectorizing it completeley, means you run out of RAM

    Examples
    --------
    >>> x, y = maps.calc_observation_grid(pixel_size=1e-6, pixel=50)
    >>> loc, vec, total = maps.get_point_sources(5000, 100)
    >>> i = randint(loc.shape[0])
    >>> calc_point_source_field(x, y, loc[i], vec[i], 5e-6)
    """
    pixel = 1#x_grid.shape[0]
    n_sources = location.shape[0]
    
    source_vector = source_vector.copy().reshape(n_sources, 1, 1, 3)
    location = location.copy().reshape(n_sources, 1, 1, 3)
    
    mx = source_vector[:, :, :, 0]
    my = source_vector[:, :, :, 1]
    mz = source_vector[:, :, :, 2]
    lx = location[:, :, :, 0]
    ly = location[:, :, :, 1]
    lz = location[:, :, :, 2] + lift_off
    
    x_grid = x_grid.reshape((1, x_grid.shape[0], -1))
    y_grid = y_grid.reshape((1, y_grid.shape[0], -1))
    
    dgridx = np.subtract(x_grid, lx)
    dgridy = np.subtract(y_grid, ly)
    
    lx = None
    ly = None
    
    squared_distance = dgridx*dgridx + dgridy*dgridy + lz*lz
    
    gridsum = mx * dgridx + my * dgridy + mz * lz 
    
    #aux = calc_loop(squared_distance,gridsum,dgridx,dgridy,lz,mx,my,mz)
    #sqrt_dist = np.sqrt(squared_distance*squared_distance*squared_distance*squared_distance*squared_distance)
    aux = np.empty(squared_distance.shape)
    
    for i in range(squared_distance.shape[0]):
        for j in range(squared_distance.shape[1]):
            for k in range(squared_distance.shape[2]):
                aux[i,j,k] = gridsum[i,j,k]/msqrt(squared_distance[i,j,k]**5)
    
    
    tmp = 1/ np.sqrt(squared_distance*squared_distance*squared_distance)
    
    squared_distance = None
    
    bx_dip = 3.0 * aux * dgridx - mx * tmp
    by_dip = 3.0 * aux * dgridy - my * tmp
    bz_dip = 3.0 * aux * lz - mz * tmp
    

    bx_tot = np.sum(bx_dip,axis=0)*9.9472e-8
    by_tot = np.sum(by_dip,axis=0)*9.9472e-8
    bz_tot = np.sum(bz_dip,axis=0)*9.9472e-8
    
    return bx_tot,by_tot,bz_tot


@numba.njit(fastmath=True,parallel=True)
def calc_loop(
        squared_distance: np.ndarray,
        gridsum: np.ndarray,
        dgridx: np.ndarray,
        dgridy: np.ndarray,
        lz: np.ndarray,
        mx: np.ndarray,
        my: np.ndarray,
        mz: np.ndarray) -> np.ndarray:
    
    aux = np.empty(squared_distance.shape)
    
    for i in numba.prange(squared_distance.shape[0]):
        for j in range(squared_distance.shape[1]):
            for k in range(squared_distance.shape[2]):
                aux[i,j,k] = gridsum[i,j,k]/msqrt(squared_distance[i,j,k]**5)
            
    #aux = gridsum / sqrt_dist
    
    return aux


@numba.jit(nopython=True, parallel=True)
def cube_jacobian(dx,interaction_l):
    """
    Build a sensitivity matrix for magnetic field of cubes around a cube
    """
    n = dx*interaction_l
        
    easting = np.arange(-n,n+.1, dx)
    northing = np.arange(-n,n+.1, dx)
    upward = np.arange(-n,n+.1, dx)

    obs= np.array([[0.,0.,0.]])
    eastingM, northingM, upwardM = np.meshgrid(easting, northing,upward)

    coordstack =np.stack([eastingM.ravel(),northingM.ravel(),upwardM.ravel()],axis=1)
    center_ind = np.argmin(np.sum(np.abs(coordstack),axis=1))

    prism_coordinates = np.stack([coordstack[:,0]-dx/2,coordstack[:,0]+dx/2, coordstack[:,1]-dx/2, coordstack[:,1]+dx/2,coordstack[:,2]-dx/2,coordstack[:,2]+dx/2],axis=1)

    jacobian_x, jacobian_y, jacobian_z = build_jacobian_choclo(obs, prism_coordinates)
            
    return jacobian_x, jacobian_y, jacobian_z


@numba.jit(nopython=True, parallel=True)
def build_jacobian_choclo(coordinates, prisms):
    """
    Build a sensitivity matrix for magnetic field of a prism
    """
    
    # make sure things are in the correct shape for our calculations
    if (coordinates.shape[0] != 3) and (coordinates.shape[1] == 3):
        coordinates = coordinates.T
        
    if (prisms.shape[1] != 6) and (prisms.shape[0] == 6):
        prisms = prisms.T
        
    # Unpack coordinates of the observation points
    easting = coordinates[0]
    northing = coordinates[1]
    upward = coordinates[2]
    
    # Initialize an empty 2d array for the sensitivity matrix
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    jacobian_x = np.empty((n_coords, n_prisms,3), dtype=np.float64)
    jacobian_y = np.empty((n_coords, n_prisms,3), dtype=np.float64)
    jacobian_z = np.empty((n_coords, n_prisms,3), dtype=np.float64)
    
    # Compute the gravity_u field that each prism generate on every observation
    # point, considering that they have a unit density
    for i in numba.prange(len(easting)):
        for j in range(prisms.shape[0]):
            jacobian_x[i, j,:] = choclo.prism.magnetic_field(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                1.0,
                0.0,
                0.0,
            )
            
            jacobian_y[i, j,:] = choclo.prism.magnetic_field(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                0.0,
                1.0,
                0.0,
            )
            
            jacobian_z[i, j,:] = choclo.prism.magnetic_field(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                0.0,
                0.0,
                1.0,
            )
    
    jacobian_x = np.nan_to_num(jacobian_x)
    jacobian_y = np.nan_to_num(jacobian_y)
    jacobian_z = np.nan_to_num(jacobian_z)   
         
    return jacobian_x, jacobian_y, jacobian_z


@numba.njit(parallel=True)
def B_cube_dict_choclo(
        observer: np.ndarray,
        polarization: np.ndarray,
        location: np.ndarray,
        dx: np.ndarray,
        ) -> np.ndarray:
    """
    Returns magnetic field components for all combinations of source and observer. Used to assemble 

    Args:
        observer (np.ndarray): _description_
        polarization (np.ndarray): _description_
        location (np.ndarray): _description_
        dx (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    # Unpack coordinates of the observation points
    easting, northing, upward = observer[:]
    
    # Initialize a result array full of zeros
    result = np.zeros((len(easting),location.shape[1],3), dtype=np.float64)
    
    # Compute the upward component that every prism generate on each
    # observation point
    for i in numba.prange(len(easting)):
        for j in range(location.shape[1]):
            result[i,j,:] += choclo.prism.magnetic_field(
                easting[i],
                northing[i],
                upward[i],
                location[0,j]-dx[j],
                location[0,j]+dx[j],
                location[1,j]-dx[j],
                location[1,j]+dx[j],
                location[2,j]-dx[j],
                location[2,j]+dx[j],
                polarization[0,j],
                polarization[1,j],
                polarization[2,j],
                )
    
    return result



@numba.njit(fastmath=True)
def B_dipole(
        observer: np.ndarray,
        polarization: np.ndarray,
        location: np.ndarray) -> np.ndarray:
    """
    Compute the field of a magnetic dipole point source

    """
    mx = polarization[:,0]
    my = polarization[:,1]
    mz = polarization[:,2]
    
    dxyz = observer - location   
    dx = dxyz[:,0]
    dy = dxyz[:,1]
    dz = dxyz[:,2]
    
    squared_distance = dx**2 + dy**2 + dz**2
    
    gridsum = mx * dx + my * dy + mz * dz
    
    aux = gridsum/np.sqrt(squared_distance**5)
    
    tmp = 1/ np.sqrt(squared_distance**3)
    
    bx = 3.0 * aux * dx - mx * tmp*9.9472e-8
    by = 3.0 * aux * dy - my * tmp*9.9472e-8
    bz = 3.0 * aux * dz - mz * tmp*9.9472e-8
    
    return bx,by,bz


# def B_dipole():
#     return

# @numba.njit(fastmath=True,parallel=True)
# def F1F2(
#     x: float,
#     y: float,
#     z: float,
#     dims: np.ndarray,
# ) -> np.ndarray:
    
#     l,w,h = dims/2      # Yang+ (1990) expression assume parallelipiped with sides of length 2l, 2w, 2h

#     # get coefficients
#     D = msqrt((l+x)**2+(h+y)**2+(w+z)**2)
    
#     F1 = atan2((h+y)*(w+z),(l+x)*D)
#     F2 = (msqrt((l+x)**2+(h+y)**2+(w-z)**2)+w-z)/(D-w-z)
    
#     return F1, F2

# def B_cuboid(
#     location: np.ndarray,
#     dims: np.ndarray,
#     polarization: np.ndarray,
# ):

#     x,y,z = location
    
#     C = MU0*polarization/(4*PI)
    
#     # x polarization
#     Bxx = -C[0] * (F1F2(-x,y,z,dims)+F1F2(-x,y,-z,dims)
    
#     return


# def magnet_cuboid_Bfield(
#     observers: np.ndarray,
#     dimensions: np.ndarray,
#     polarizations: np.ndarray,
# ):
#     """B-field of homogeneously magnetized cuboids in Cartesian Coordinates.

#     The cuboids sides are parallel to the coordinate axes. The geometric centers of the
#     cuboids lie in the origin. The output is proportional to the polarization magnitude
#     and independent of the length units chosen for observers and dimensions.

#     Parameters
#     ----------
#     observers: ndarray, shape (n,3)
#         Observer positions (x,y,z) in Cartesian coordinates.

#     dimensions: ndarray, shape (n,3)
#         Length of cuboids sides.

#     polarizations: ndarray, shape (n,3)
#         Magnetic polarization vectors.

#     Returns
#     -------
#     B-Field: ndarray, shape (n,3)
#         B-field generated by Cuboids at observer positions.

#     Notes
#     -----
#     Field computations via magnetic surface charge density. Published
#     several times with similar expressions:

#     Yang: Superconductor Science and Technology 3(12):591 (1990)

#     Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4 (2005)

#     Camacho: Revista Mexicana de Fisica E 59 (2013) 8-17

#     Avoiding indeterminate forms:

#     In the above implementations there are several indeterminate forms
#     where the limit must be taken. These forms appear at positions
#     that are extensions of the edges in all xyz-octants except bottQ4.
#     In the vicinity of these indeterminate forms the formula becomes
#     numerically instable.

#     Chosen solution: use symmetries of the problem to change all
#     positions to their bottQ4 counterparts. see also

#     Cichon: IEEE Sensors Journal, vol. 19, no. 7, April 1, 2019, p.2509
#     """
#     pol_x, pol_y, pol_z = polarizations.T
#     a, b, c = dimensions.T / 2
#     x, y, z = np.copy(observers).T

#     # avoid indeterminate forms by evaluating in bottQ4 only --------
#     # basic masks
#     maskx = x < 0
#     masky = y > 0
#     maskz = z > 0

#     # change all positions to their bottQ4 counterparts
#     x[maskx] = x[maskx] * -1
#     y[masky] = y[masky] * -1
#     z[maskz] = z[maskz] * -1

#     # create sign flips for position changes
#     qsigns = np.ones((len(pol_x), 3, 3))
#     qs_flipx = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
#     qs_flipy = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
#     qs_flipz = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
#     # signs flips can be applied subsequently
#     qsigns[maskx] = qsigns[maskx] * qs_flipx
#     qsigns[masky] = qsigns[masky] * qs_flipy
#     qsigns[maskz] = qsigns[maskz] * qs_flipz

#     # field computations --------------------------------------------
#     # Note: in principle the computation for all three polarization-components can be
#     #   vectorized itself using symmetries. However, tiling the three
#     #   components will cost more than is gained by the vectorized evaluation

#     # Note: making the following computation steps is not necessary
#     #   as mkl will cache such small computations
#     xma, xpa = x - a, x + a
#     ymb, ypb = y - b, y + b
#     zmc, zpc = z - c, z + c

#     xma2, xpa2 = xma**2, xpa**2
#     ymb2, ypb2 = ymb**2, ypb**2
#     zmc2, zpc2 = zmc**2, zpc**2

#     mmm = np.sqrt(xma2 + ymb2 + zmc2)
#     pmp = np.sqrt(xpa2 + ymb2 + zpc2)
#     pmm = np.sqrt(xpa2 + ymb2 + zmc2)
#     mmp = np.sqrt(xma2 + ymb2 + zpc2)
#     mpm = np.sqrt(xma2 + ypb2 + zmc2)
#     ppp = np.sqrt(xpa2 + ypb2 + zpc2)
#     ppm = np.sqrt(xpa2 + ypb2 + zmc2)
#     mpp = np.sqrt(xma2 + ypb2 + zpc2)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         ff2x = np.log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp)) - np.log(
#             (xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp)
#         )

#         ff2y = np.log(
#             (-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp)
#         ) - np.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

#         ff2z = np.log(
#             (-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp)
#         ) - np.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

#     ff1x = (
#         np.arctan2((ymb * zmc), (xma * mmm))
#         - np.arctan2((ymb * zmc), (xpa * pmm))
#         - np.arctan2((ypb * zmc), (xma * mpm))
#         + np.arctan2((ypb * zmc), (xpa * ppm))
#         - np.arctan2((ymb * zpc), (xma * mmp))
#         + np.arctan2((ymb * zpc), (xpa * pmp))
#         + np.arctan2((ypb * zpc), (xma * mpp))
#         - np.arctan2((ypb * zpc), (xpa * ppp))
#     )

#     ff1y = (
#         np.arctan2((xma * zmc), (ymb * mmm))
#         - np.arctan2((xpa * zmc), (ymb * pmm))
#         - np.arctan2((xma * zmc), (ypb * mpm))
#         + np.arctan2((xpa * zmc), (ypb * ppm))
#         - np.arctan2((xma * zpc), (ymb * mmp))
#         + np.arctan2((xpa * zpc), (ymb * pmp))
#         + np.arctan2((xma * zpc), (ypb * mpp))
#         - np.arctan2((xpa * zpc), (ypb * ppp))
#     )

#     ff1z = (
#         np.arctan2((xma * ymb), (zmc * mmm))
#         - np.arctan2((xpa * ymb), (zmc * pmm))
#         - np.arctan2((xma * ypb), (zmc * mpm))
#         + np.arctan2((xpa * ypb), (zmc * ppm))
#         - np.arctan2((xma * ymb), (zpc * mmp))
#         + np.arctan2((xpa * ymb), (zpc * pmp))
#         + np.arctan2((xma * ypb), (zpc * mpp))
#         - np.arctan2((xpa * ypb), (zpc * ppp))
#     )

#     # contributions from x-polarization
#     #    the 'missing' third sign is hidden in ff1x
#     bx_pol_x = pol_x * ff1x * qsigns[:, 0, 0]
#     by_pol_x = pol_x * ff2z * qsigns[:, 0, 1]
#     bz_pol_x = pol_x * ff2y * qsigns[:, 0, 2]
#     # contributions from y-polarization
#     bx_pol_y = pol_y * ff2z * qsigns[:, 1, 0]
#     by_pol_y = pol_y * ff1y * qsigns[:, 1, 1]
#     bz_pol_y = -pol_y * ff2x * qsigns[:, 1, 2]
#     # contributions from z-polarization
#     bx_pol_z = pol_z * ff2y * qsigns[:, 2, 0]
#     by_pol_z = -pol_z * ff2x * qsigns[:, 2, 1]
#     bz_pol_z = pol_z * ff1z * qsigns[:, 2, 2]

#     # summing all contributions
#     bx_tot = bx_pol_x + bx_pol_y + bx_pol_z
#     by_tot = by_pol_x + by_pol_y + by_pol_z
#     bz_tot = bz_pol_x + bz_pol_y + bz_pol_z

#     # B = np.c_[bx_tot, by_tot, bz_tot]      # faster for 10^5 and more evaluations
#     B = np.concatenate(((bx_tot,), (by_tot,), (bz_tot,)), axis=0).T

#     B /= 4 * np.pi
#     return B

# ''' MAGNETIC FIELD CALCULATIONS FROM MATPYLIB'''


# def dipole_Bfield(
#     observers: np.ndarray,
#     moments: np.ndarray,
# ) -> np.ndarray:
#     """Magnetic field of a dipole moments.

#     The dipole moment lies in the origin of the coordinate system.
#     The output is proportional to the moment input, and is independent
#     of length units used for observers (and moment) input considering
#     that the moment is proportional to [L]**2.
#     Returns np.inf for all non-zero moment components in the origin.

#     Parameters
#     ----------
#     observers: ndarray, shape (n,3)
#         Observer positions (x,y,z) in Cartesian coordinates.

#     moments: ndarray, shape (n,3)
#         Dipole moment vector.

#     Returns
#     -------
#     H-field: ndarray, shape (n,3)
#         H-field of Dipole in Cartesian coordinates.

#     Notes
#     -----
#     The moment of a magnet is given by its volume*magnetization.
#     """

#     x, y, z = observers.T
#     r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
#     with np.errstate(divide="ignore", invalid="ignore"):
#         # 0/0 produces invalid warn and results in np.nan
#         # x/0 produces divide warn and results in np.inf
#         H = (
#             (
#                 3 * np.sum(moments * observers, axis=1) * observers.T / r**5
#                 - moments.T / r**3
#             ).T
#             / 4
#             / np.pi
#         )

#     # when r=0 return np.inf in all non-zero moments directions
#     mask1 = r == 0
#     if np.any(mask1):
#         with np.errstate(divide="ignore", invalid="ignore"):
#             H[mask1] = moments[mask1] / 0.0
#             np.nan_to_num(H, copy=False, posinf=np.inf, neginf=-np.inf)

#     return H * MU0


# def magnet_cuboid_Bfield(
#     observers: np.ndarray,
#     dimensions: np.ndarray,
#     polarizations: np.ndarray,
# ):
#     """B-field of homogeneously magnetized cuboids in Cartesian Coordinates.

#     The cuboids sides are parallel to the coordinate axes. The geometric centers of the
#     cuboids lie in the origin. The output is proportional to the polarization magnitude
#     and independent of the length units chosen for observers and dimensions.

#     Parameters
#     ----------
#     observers: ndarray, shape (n,3)
#         Observer positions (x,y,z) in Cartesian coordinates.

#     dimensions: ndarray, shape (n,3)
#         Length of cuboids sides.

#     polarizations: ndarray, shape (n,3)
#         Magnetic polarization vectors.

#     Returns
#     -------
#     B-Field: ndarray, shape (n,3)
#         B-field generated by Cuboids at observer positions.

#     Notes
#     -----
#     Field computations via magnetic surface charge density. Published
#     several times with similar expressions:

#     Yang: Superconductor Science and Technology 3(12):591 (1990)

#     Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4 (2005)

#     Camacho: Revista Mexicana de Fisica E 59 (2013) 8-17

#     Avoiding indeterminate forms:

#     In the above implementations there are several indeterminate forms
#     where the limit must be taken. These forms appear at positions
#     that are extensions of the edges in all xyz-octants except bottQ4.
#     In the vicinity of these indeterminate forms the formula becomes
#     numerically instable.

#     Chosen solution: use symmetries of the problem to change all
#     positions to their bottQ4 counterparts. see also

#     Cichon: IEEE Sensors Journal, vol. 19, no. 7, April 1, 2019, p.2509
#     """
#     pol_x, pol_y, pol_z = polarizations.T
#     a, b, c = dimensions.T / 2
#     x, y, z = np.copy(observers).T

#     # avoid indeterminate forms by evaluating in bottQ4 only --------
#     # basic masks
#     maskx = x < 0
#     masky = y > 0
#     maskz = z > 0

#     # change all positions to their bottQ4 counterparts
#     x[maskx] = x[maskx] * -1
#     y[masky] = y[masky] * -1
#     z[maskz] = z[maskz] * -1

#     # create sign flips for position changes
#     qsigns = np.ones((len(pol_x), 3, 3))
#     qs_flipx = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
#     qs_flipy = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
#     qs_flipz = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
#     # signs flips can be applied subsequently
#     qsigns[maskx] = qsigns[maskx] * qs_flipx
#     qsigns[masky] = qsigns[masky] * qs_flipy
#     qsigns[maskz] = qsigns[maskz] * qs_flipz

#     # field computations --------------------------------------------
#     # Note: in principle the computation for all three polarization-components can be
#     #   vectorized itself using symmetries. However, tiling the three
#     #   components will cost more than is gained by the vectorized evaluation

#     # Note: making the following computation steps is not necessary
#     #   as mkl will cache such small computations
#     xma, xpa = x - a, x + a
#     ymb, ypb = y - b, y + b
#     zmc, zpc = z - c, z + c

#     xma2, xpa2 = xma**2, xpa**2
#     ymb2, ypb2 = ymb**2, ypb**2
#     zmc2, zpc2 = zmc**2, zpc**2

#     mmm = np.sqrt(xma2 + ymb2 + zmc2)
#     pmp = np.sqrt(xpa2 + ymb2 + zpc2)
#     pmm = np.sqrt(xpa2 + ymb2 + zmc2)
#     mmp = np.sqrt(xma2 + ymb2 + zpc2)
#     mpm = np.sqrt(xma2 + ypb2 + zmc2)
#     ppp = np.sqrt(xpa2 + ypb2 + zpc2)
#     ppm = np.sqrt(xpa2 + ypb2 + zmc2)
#     mpp = np.sqrt(xma2 + ypb2 + zpc2)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         ff2x = np.log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp)) - np.log(
#             (xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp)
#         )

#         ff2y = np.log(
#             (-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp)
#         ) - np.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

#         ff2z = np.log(
#             (-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp)
#         ) - np.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

#     ff1x = (
#         np.arctan2((ymb * zmc), (xma * mmm))
#         - np.arctan2((ymb * zmc), (xpa * pmm))
#         - np.arctan2((ypb * zmc), (xma * mpm))
#         + np.arctan2((ypb * zmc), (xpa * ppm))
#         - np.arctan2((ymb * zpc), (xma * mmp))
#         + np.arctan2((ymb * zpc), (xpa * pmp))
#         + np.arctan2((ypb * zpc), (xma * mpp))
#         - np.arctan2((ypb * zpc), (xpa * ppp))
#     )

#     ff1y = (
#         np.arctan2((xma * zmc), (ymb * mmm))
#         - np.arctan2((xpa * zmc), (ymb * pmm))
#         - np.arctan2((xma * zmc), (ypb * mpm))
#         + np.arctan2((xpa * zmc), (ypb * ppm))
#         - np.arctan2((xma * zpc), (ymb * mmp))
#         + np.arctan2((xpa * zpc), (ymb * pmp))
#         + np.arctan2((xma * zpc), (ypb * mpp))
#         - np.arctan2((xpa * zpc), (ypb * ppp))
#     )

#     ff1z = (
#         np.arctan2((xma * ymb), (zmc * mmm))
#         - np.arctan2((xpa * ymb), (zmc * pmm))
#         - np.arctan2((xma * ypb), (zmc * mpm))
#         + np.arctan2((xpa * ypb), (zmc * ppm))
#         - np.arctan2((xma * ymb), (zpc * mmp))
#         + np.arctan2((xpa * ymb), (zpc * pmp))
#         + np.arctan2((xma * ypb), (zpc * mpp))
#         - np.arctan2((xpa * ypb), (zpc * ppp))
#     )

#     # contributions from x-polarization
#     #    the 'missing' third sign is hidden in ff1x
#     bx_pol_x = pol_x * ff1x * qsigns[:, 0, 0]
#     by_pol_x = pol_x * ff2z * qsigns[:, 0, 1]
#     bz_pol_x = pol_x * ff2y * qsigns[:, 0, 2]
#     # contributions from y-polarization
#     bx_pol_y = pol_y * ff2z * qsigns[:, 1, 0]
#     by_pol_y = pol_y * ff1y * qsigns[:, 1, 1]
#     bz_pol_y = -pol_y * ff2x * qsigns[:, 1, 2]
#     # contributions from z-polarization
#     bx_pol_z = pol_z * ff2y * qsigns[:, 2, 0]
#     by_pol_z = -pol_z * ff2x * qsigns[:, 2, 1]
#     bz_pol_z = pol_z * ff1z * qsigns[:, 2, 2]

#     # summing all contributions
#     bx_tot = bx_pol_x + bx_pol_y + bx_pol_z
#     by_tot = by_pol_x + by_pol_y + by_pol_z
#     bz_tot = bz_pol_x + bz_pol_y + bz_pol_z

#     # B = np.c_[bx_tot, by_tot, bz_tot]      # faster for 10^5 and more evaluations
#     B = np.concatenate(((bx_tot,), (by_tot,), (bz_tot,)), axis=0).T

#     B /= 4 * np.pi
#     return B