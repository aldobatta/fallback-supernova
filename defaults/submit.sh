#!/bin/bash
#
# SLURM resource specifications
# (use an extra '#' in front of SBATCH to comment-out any unused options)
#
#SBATCH --job-name=test_1   # shows up in the output of 'squeue'
#SBATCH --time=4-00:00:00       # specify the requested wall-time
#SBATCH --partition=dark  # specify the partition to run on
#SBATCH --nodes=4               # number of nodes allocated for this job
#SBATCH --ntasks-per-node=1    # number of MPI ranks per node
#SBATCH --cpus-per-task=32       # number of OpenMP threads per MPI rank
##SBATCH --exclude=<node list>  # avoid nodes (e.g. --exclude=node786)
#SBATCH -o test_out                                                                                                                                          
#SBATCH -e test_err    


# Load default settings for environment variables
source /users/software/astro/startup.d/modules.sh

# If required, replace specific modules
# module unload intelmpi
# module load mvapich2

module load intelmpi/5.0.2.044 intel/15.0.1

# When compiling remember to use the same environment and modules


EXE="/groups/dark/avigna/gadget/Gadget2/ModGadget/Timestep"
### ARGS is optional. If you don't need it simply leave it empty. YOU ALWAYS NEEd THIS ONE                   
ARGS="dir/paramfile.tex"
### INFILE = 1 when restarting gadget from restart files, nothing when you start with IC                     
# AVG: 2 if IC is in an HDF5 format?
INFILE="2"

# Execute the code
srun  $EXE $ARGS $INFILE
