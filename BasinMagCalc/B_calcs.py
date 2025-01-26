
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
    
    def B_calc_cuboid(self):
        return
    
    def make_B_dict_cubes(self, dist_crit):
        '''
        calculate the normalized cuboid field components for all combos of size and direction that can occur in 
        the tree structure.
        
        comb_crit: Maximum number of cell combination iterations
        dist_crit: Maximum distance for cells to be treated as near-field
        '''

        # all unique sizes of root cells
        cell_lens, _ = self.get_cell_dxs()
        
        # if cells are not all the same size, 
        if len(cell_lens) > 1:
            comb_crit = np.max(cell_lens)
        
        # get maximum number of cells of each size which would be considered near-field
        n_crit = np.ceil(dist_crit/cell_lens)
        
        
        for i,cell_size in enumerate(cell_lens):
            dists_1 = np.append(dists_1,np.repeat(cell_size,n_crit[i]))
            
        # all possible combinations of displacements to near field cells
        Tm=np.matmul(dists_1.reshape(-1,1)/2,np.ones((1,len(dists_1))))
        dists = np.unique(Tm+Tm.T)
        
        # drop out combinations that are much larger than critical distance
        dists = dists[np.floor(dists/dist_crit)>1]
        
        lx, ly, lz = np.meshgrid([dists,dists,dists])
        
        observers = 
        
        B_cubes_x = B_cube_dict_choclo(observers,[1,0,0],np.zeros(len),cell_lens)
        B_cubes_y = B_cube_dict_choclo(observers,[0,1,0],np.zeros(len),cell_lens)
        B_cubes_z = B_cube_dict_choclo(observers,[0,0,1],np.zeros(len),cell_lens)
        
        return



    
    def B_calc_tree(self,itrp,particles,Bvec,epsilon=1e-6):
        
        # loop over particles and drop out any where M is small
        for i, particle in enumerate(self.particles):
            # initialize list of particles
            particles_i = []
        
            # get delta M from radial and depth positions
            delM = itrp(np.sqrt(particle.x**2+particle.y**2),particle.z**2)
            
            # if the total magnetization is large enough, add to list of particles
            if np.abs(particle.M) >= epsilon:
                
                # calculate updated multipole coefficients
                
                particles.append(particle)
                
                
                
                
            # if delta M is large enough, add to list of particles
            elif np.abs(delM) >= epsilon:
                particles.append(particle)
            
            
            
        return
    
    def B_calc_new(self,Bvec,epsilon=1e-6):
        
        ## setup tasks
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        # flatten r map array
        rinds = self.rmap.flatten()
        
        # sort array, make recovery index
        sorted_r = np.sort(rinds)
        recovinds_r = np.argsort(np.argsort(rinds))
        
        # make particles everywhere the total magnetization is greater than epsilon
        self.make_Particles(epsilon=epsilon)
        
        # set magnetization from first time step
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        ## loop over time steps
        for ti, t in enumerate(self.t_list[1:]):
            # create interpolations
            itrp_dM = interp2d(self.xvals,self.zvals,self.dM_mat[ti+1,:,:]) # MAKE TIME DEPENDENT

            self.B_calc_tree(itrp_dM,self.particles,Bvec,epsilon=epsilon)
            
            # make position arrays for dipole_sources function
            xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
            xyz_grid = np.meshgrid(xv_whole,self.zvals,xv_whole)
            
            zm_list = xyz_grid[1]
            xm_list = xyz_grid[0]
            ym_list = xyz_grid[2]
            
            
        return


    def Bcalc_local(self, mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off, z_lim=0):
        
        
        limx = imgrid[0].shape[0]
        limy = imgrid[0].shape[1]
        
        
            
        # make magnetic moment array
        mag_new = mag_moms[zi,:,:].reshape((-1,1))
        #mag_old = (mag_moms[zi,:,:].reshape((-1,1))-1)*np.array(og_vec).reshape((1,-1))*self.cellsize**3*self.susc
        mag_xyz =  mag_new #+ mag_old
        
        zero_inds = np.sum(mag_xyz**2,axis=1)**2>0.0001**2
        
        mag_xyz = mag_xyz[zero_inds,:]*np.array(self.mag_vec).reshape((1,-1))*self.cellsize**3*self.susc
        
        xl_z = xm_list[zi,:,:].reshape((-1,1))[zero_inds,:]
        yl_z = ym_list[zi,:,:].reshape((-1,1))[zero_inds,:]
        zl_z = np.abs(zm_list[zi,:,:].reshape((-1,1)))[zero_inds,:]
        
    
        loc_xyz = np.concatenate((xl_z,yl_z,zl_z),axis=1)
        
        t0 = time.time()
        
        # calculate magnetic fields 
        Bmapi = calc_all(imgrid[0],imgrid[1],loc_xyz[:,:],mag_xyz[:,:],lift_off*1000)
        #print('Mag calculation time: %1.3f'%(time.time() - t0))
        #calc_point_source_field.parallel_diagnostics(level=4)
        Bmapx += Bmapi[0]
        Bmapy += Bmapi[1]
        Bmapz += Bmapi[2]
             
        # put into nT (double check these units)
        Bmapx = Bmapx*1e9
        Bmapy = Bmapy*1e9
        Bmapz = Bmapz*1e9
        
        print([np.max(np.abs(Bmapx)),np.max(np.abs(Bmapy)),np.max(np.abs(Bmapz))])
        
        # calculate field magnitude
        Bmap = np.sqrt(Bmapx**2+Bmapy**2+Bmapz**2)
        
        print(np.max(np.abs(Bmap)))
        
        return  Bmap, Bmapz
    
    
    def Bcalc(self, mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off, z_lim=0):

        limx = imgrid[0].shape[0]
        limy = imgrid[0].shape[1]
        
        Bmapx = np.zeros((limx,limy))
        Bmapy = np.zeros((limx,limy))
        Bmapz = np.zeros((limx,limy))
        
        # loop over depths and calculate magnetic field
        for zi in range(z_lim,len(self.zvals)):
            
            if zi%10==0:
                print(zi)
                
            
            
            # make magnetic moment array
            mag_new = mag_moms[zi,:,:].reshape((-1,1))
            #mag_old = (mag_moms[zi,:,:].reshape((-1,1))-1)*np.array(og_vec).reshape((1,-1))*self.cellsize**3*self.susc
            mag_xyz =  mag_new #+ mag_old
            
            zero_inds = np.sum(mag_xyz**2,axis=1)**2>0.0001**2
            
            mag_xyz = mag_xyz[zero_inds,:]*np.array(self.mag_vec).reshape((1,-1))*self.cellsize**3*self.susc
            
            xl_z = xm_list[zi,:,:].reshape((-1,1))[zero_inds,:]
            yl_z = ym_list[zi,:,:].reshape((-1,1))[zero_inds,:]
            zl_z = np.abs(zm_list[zi,:,:].reshape((-1,1)))[zero_inds,:]
            
        
            loc_xyz = np.concatenate((xl_z,yl_z,zl_z),axis=1)
            
            t0 = time.time()
           
            # calculate magnetic fields 
            Bmapi = calc_all(imgrid[0],imgrid[1],loc_xyz[:,:],mag_xyz[:,:],lift_off*1000)
            #print('Mag calculation time: %1.3f'%(time.time() - t0))
            #calc_point_source_field.parallel_diagnostics(level=4)
            Bmapx += Bmapi[0]
            Bmapy += Bmapi[1]
            Bmapz += Bmapi[2]
             
            
            
        # put into nT
        Bmapx = Bmapx*1e9
        Bmapy = Bmapy*1e9
        Bmapz = Bmapz*1e9
        
        print([np.max(np.abs(Bmapx)),np.max(np.abs(Bmapy)),np.max(np.abs(Bmapz))])
        
        # calculate field magnitude
        Bmap = np.sqrt(Bmapx**2+Bmapy**2+Bmapz**2)
        
        print(np.max(np.abs(Bmap)))
        
        return  Bmap, Bmapz
    
    
    def Bnet(self,quadr_mag_frac,mag_old,lift_off,imgrid=np.meshgrid(np.arange(-100e3,100e3,10e3),np.arange(-100e3,100e3,10e3)),manualcalc=True):
        '''
        Makes map of magnetic field over specific magnetization ensemble at a given measurement height
        
        quadr_mag_frac: magnetization intensity for one quadrant
        mag_vec: magnetization direction vector
        lift_off: measurement height (km)
        '''
        
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        limx = imgrid[0].shape[0]
        limy = imgrid[0].shape[1]
        
        # make magnetic moments matrix
        mag_moms = quadr_mag_frac[::-1,:,:]
        mag_moms = np.concatenate((mag_moms[:,:0:-1,:],mag_moms),axis=1)
        mag_moms = np.concatenate((mag_moms[:,:,:0:-1],mag_moms),axis=2)
        
        # make position arrays for dipole_sources function
        xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
        xyz_grid = np.meshgrid(xv_whole,self.zvals,xv_whole)
        
        zm_list = xyz_grid[1]
        xm_list = xyz_grid[0]
        ym_list = xyz_grid[2]
        
        Bmapx = np.zeros((limx,limy))
        Bmapy = np.zeros((limx,limy))
        Bmapz = np.zeros((limx,limy))
        
        # add background magnetization
        print('Adding random background magnetization')
        # if self.bg_mag:
        #
        #     max_dm = self.make_old_mag()
        #
        #     # calculate background fields
        #     rand_bg = BGLil10(xm_list, zm_list, sat_frac=1)
        #     mag_moms = mag_moms + rand_bg*(1-max_dm)
        
        # add impact layer
        print('Adding late impacts')
        
        
        mag_moms[self.lr_out>0] = 0
        
        # calculate field including late impacts
        Bmap, Bmapz = self.Bcalc(mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off)
        
        
        return  Bmap, Bmapz, mag_moms 