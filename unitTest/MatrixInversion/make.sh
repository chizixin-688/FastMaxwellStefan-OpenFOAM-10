source /public1/soft/modules/module.sh      
module load mpi/oneAPI/2022.1 gcc/7.3.0 cmake/4.2.0
source /public1/home/sch9617/soft/OpenFOAM-10/etc/bashrc
g++ InversionPerformance.C LUsolver.C LUxSolve.C -O3 -mavx2  -mfma -std=c++14 \
-I${MKLROOT}/include \
-I$FOAM_SRC/OpenFOAM/lnInclude \
-I$FOAM_SRC/finiteVolume/lnInclude \
-L${MKLROOT}/lib/intel64 \
-L$FOAM_LIBBIN -lOpenFOAM -lfiniteVolume \
-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread \
-liomp5 -lpthread -lm -ldl -o test_inv
echo "Compilation complete. Executable is 'test_inv'."
