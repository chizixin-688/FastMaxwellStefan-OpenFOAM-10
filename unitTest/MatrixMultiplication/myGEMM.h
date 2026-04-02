// these function are only suitable for square matrix

#ifndef MY_GEMM_H
#define MY_GEMM_H


#include <immintrin.h>


inline double get_elem0(__m256d vec);
inline double get_elem0(__m128d vec);
inline double get_elem1(__m256d vec);
inline double get_elem1(__m128d vec);
inline double get_elem2(__m256d vec);
inline double get_elem3(__m256d vec);

void MatrixMultiply484_0
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_1
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_2
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_3
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);
void MatrixMultiply484_4
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_5
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_6
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

void MatrixMultiply484_7
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
);

#endif
