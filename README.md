# hpc-with-python
Implementation of the Lattice Boltzmann Method. Includes simulations of streaming, collision, and shear-wave decay; visualizations for Couette and Poiseuille Flow, and a parallelized two-dimensional version of the Sliding Lid experiment performed on the BWUniCluster 2.0

* shear wave decay
- can be run as is

* couette flow
- can be run as is

poiseuille flow
- Can be run as is

* sliding_lid
- serial version of the sliding lid experiment
- Can be run as is

* Sliding_lid_parallelized
-local parallelized version of the sliding lid experiment
- ux.npy and uy.npy generated as a result of running it
- to visualize results, run ./BWUniCluster/plot.py and set:
	u = np.load('ux.npy')
	v = np.load('uy.npy')

* ./BWUniCluster/sliding_lid_parallel.py
- to run on BWUniCluster 2.0, upload all files in ./BWUniCluster/MLUPS/sh as well as sliding_lid_parallel.py
- to run for 300x300 or 500x500 grid, within sliding_lid_parallel.py, change grid_x and grid_y to those required
 and also change parameters of those mentioned in the caption of 4.8 of the report, in order simulate for Re=1000
- .out files are generated
- results from previous simulations are stored for the 300x300 and 500x500 versions in folders ./BWUniCluster/MLUPS/300x300
and ./BWUniCluster/MLUPS/500x500 respectively
- to visualize results, .npy files of one simulation for each grid size has been saved
- to plot, run ./BWUniCluster/plot.py and set:
	u = np.load('ux_300.npy')   	 u = np.load('ux_500.npy')
	v = np.load('uy_300.npy')   OR   v = np.load('uy_500.npy')
 depending on the grid size desired

* plot MLUPS
- run ./BWUniCluster/MLUPS/plot_MLUPS.py
- values plotted are copied over from the .out files within folders ./BWUniCluster/MLUPS/300x300 ./BWUniCluster/MLUPS/500x500
	
 

