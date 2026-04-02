#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p v6_384
#SBATCH --exclusive
source /public1/soft/modules/module.sh
module load mpi/oneAPI/2022.1 gcc/7.3.0
source /public1/home/sch9617/soft/OpenFOAM-10/etc/bashrc
#source /public1/home/sch9617/soft/mmcFoam/etc/bashrc
export MKL_DEBUG_CPU_TYPE=5 
./test_inv
