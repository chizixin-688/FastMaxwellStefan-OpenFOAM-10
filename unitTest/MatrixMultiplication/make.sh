g++ GEMMPerformance.C myGEMM.C -O3 -mavx2  -mfma -march=native -std=c++14 \
-I${MKLROOT}/include \
-L${MKLROOT}/lib/intel64 \
-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread \
-liomp5 -lpthread -lm -ldl -o test_gemm
echo "Compilation complete. Executable is 'test_gemm'."
