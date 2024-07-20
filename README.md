# BasinCoolMag
## Reproduction code for: Could the weak magnetism of Martian impact basins reflect cooling in a reversing dynamo?

This code is divided into two components: cooling calculations (BasinCooling) and magnetic field calculations (BasinMagCalc). 

## BasinCooling

### System requirements:
Tested on Ubuntu 20.04.06 (Focal Fossa)

- C++ 
- gcc
- CMake
- GNU Make
- Gmsh
- Armadillo (and its dependencies)
- deal.ii >=9 (installed with libconfig++-dev, BLAS, LAPACK, UMFPACK, and Gmsh)

It is also recommended to install ParaView or a similar software for viewing .inp and .vtk files. The meshio package in Python is also a good option; an example of the implementation is included in read_heat_steps.ipynb.

### Installing deal.II and dependencies:

First, make sure that libconfig++-dev, BLAS, LAPACK, UMFPACK, Gmsh, and Armadillo (and its dependencies) are installed. I used the commands:

```
sudo apt install cmake
sudo apt install libconfig++-dev
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install libsuitesparse-dev
sudo apt-get install libarpack2-dev
sudo apt-get install libsuperlu-dev
sudo apt-get install libopenblas-openmp-dev
sudo apt install gmsh
```
and installed Armadillo from the download available at https://arma.sourceforge.net/download.html.

Then download deal.ii from https://www.dealii.org/download.html and follow the installation instructions. To install deal.II v9.5.2, I used the commands:

```
gunzip deal.II-9.5.2.tar.gz
tar xf deal.II-9.5.2.tar
mkdir diibuild
cd diibuild
cmake -DCMAKE_INSTALL_PREFIX=/lib/dealii ../dealii-9.5.2 -DDEAL_II_WITH_GMSH=ON
sudo make –jobs=8 install
```

CMake should automatically find and use the extra libraries (e.g. BLAS, LAPACK, Gmsh). If that fails, you may need to follow the more detailed instructions on the deal.II website to install everything necessary.


### Instructions:

Use Preprocessing/CoolingNotebook.ipynb to produce configuration files for analytical initial temperature conditions OR download configuration files (config.cfg, mesh.*, T_smooth.txt, eq_T_smooth.txt, xc.txt, zc.txt [, kappa.txt]) for the desired basin size from the Harvard Dataverse (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TPW4WB&faces-redirect=true). These files should be stored in a dedicated folder which must contain a folder named "output" for run outputs. Check that file paths in config.cfg match the corresponding files. 

Change the run target in mars_heat.cc main, then compile and run the code. I used the commands:

```
cmake . -DDEAL_II_DIR=/lib/dealii
make run
```

The notebook read_heat_steps.ipynb includes simple scripts for viewing cooling data.

## BasinMagCalc

### System requirements:
Tested on Windows 10 22H2 and Rocky Linux 8.7 (Green Obsidian)

Python 3.8.3-3.10.9
Required Python packages:  
numpy, pandas, matplotlib, time, os, numba, meshio, asyncio, joblib, scipy<1.14.0


### Installation:
Create a new Conda/Mamba/etc. environment and install the required Python packages. 
	
Estimated install time: 20 minutes


### Demo instructions:
Running basiniter.py as-is from within the BasinMagCalc folder will perform the magnetic field calculations described in the manuscript Methods section over an 80x2500km region at 200 km altitude for a 600 km demo basin. The included demo will test 5 random reversal histories and save the output in \600km\mag_output\. 
	
Running this code will output:
1. BMaps: Final magnetic field maps
2. BMaps_nolr: Magnetic field maps neglecting late remagnetization
3. RevRates: Mean reversal rates
4. Revs: Full reversal histories

Expected output is included in \600km\mag_output_expected\. 
	
Estimated run time: 15 minutes
	
Also included are some simple analysis tools in mag_data_analysis.ipynb. Note that the magnetization intensity assumed in the magnetic field calculation scripts is 1 A/m; since magnetic field strength scales linearly with magnetization intensity, one can multiply final magnetic field maps by the desired amount to compare to manuscript values.
	
	
### Further use/reproduction instructions:
Any or all data in the manuscript can be reproduced by downloading thermal histories for different basin sizes from the Harvard Dataverse (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TPW4WB&faces-redirect=true) and correspondingly changing input/output directories in basiniter.py. Different reversal frequencies can be tested by changing the mu and thresh parameters in the do_basiniter_dual() function call. The field mapping domain can be changed by defining a new imgrid or choosing one of the different supplied imgrids.

Note: performing these calculations for larger basins, or for larger imgrids, can take many hours and be VERY resource intensive. Large-domain mapping for the largest basins typically uses ~200 GB memory! Be cautious attempting calculations for any basin sizes larger than 600-800km diameter on a normal desktop. 
	