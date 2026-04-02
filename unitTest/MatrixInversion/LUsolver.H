/*---------------------------------------------------------------------------*\
  Copyright (C) 2026 Zixin Chi
  All rights reserved.

  This file is an original work developed independently by Zixin Chi.

  This program is distributed under the terms of the
  GNU General Public License v3.0.

  Description
      This file is used for LU decomposition and matrix inversion

  Author
      Zixin Chi <chizixin@buaa.edu.cn>

\*---------------------------------------------------------------------------*/

//---------------------------------
// 1. Standard C++ library headers
//---------------------------------
#include <vector>

//---------------------------------
// 2. FastChemistry headers
//---------------------------------
//#include "Macro.H"

//---------------------------------
// 3. SIMD / AVX2 headers
//---------------------------------
#include <immintrin.h>

//=============================================================================//

#ifndef LUsolver_H
#define LUsolver_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


namespace Foam
{

class LUsolver
{
    // Private data

        //- Matrix size
        unsigned int N;

        unsigned int alignN;

        //- Remain
        unsigned int Remain;

        //- Data pointer
        double* __restrict__ v_;

        //- Store the global index of target row.
        unsigned int* pivotIndice_;

        //- Store the row address
        std::vector<double*> rowPtr;

        //- Save the reciprocal of diagonal elements
        std::vector<double> invD;

        //- Pointer array pointing each row of matrix
        // for inverse matrix
        std::vector<double*> invMatRowPtr;

        double LuLimiter = 2.220446049250313e-16;

        using UpdateFunc = void (LUsolver::*)(unsigned int, unsigned int, unsigned int, unsigned int);

        UpdateFunc updateTable[3][2] =
        {
            {   
                &LUsolver::UpdateL22U22_Vec_38_00,
                &LUsolver::UpdateL22U22_Vec_38_04
            },
            {   
                &LUsolver::UpdateL22U22_Vec_38_10,
                &LUsolver::UpdateL22U22_Vec_38_14
            },
            {
                &LUsolver::UpdateL22U22_Vec_38_20,
                &LUsolver::UpdateL22U22_Vec_38_24
            }
        };


        using invMatFunc = void (LUsolver::*)(double* __restrict__, const int, const int );

        invMatFunc invMatFuncTable[14] =
        {
            &LUsolver::invMatrix56,
            &LUsolver::invMatrix52,
            &LUsolver::invMatrix48,
            &LUsolver::invMatrix44,
            &LUsolver::invMatrix40,
            &LUsolver::invMatrix36,
            &LUsolver::invMatrix32,
            &LUsolver::invMatrix28,
            &LUsolver::invMatrix24,
            &LUsolver::invMatrix20,
            &LUsolver::invMatrix16,
            &LUsolver::invMatrix12,
            &LUsolver::invMatrix8,
            &LUsolver::invMatrix4,
        };


public:

        int callTable[14] = {
            0,0,0,0,0,0,0,
            0,0,0,0,0,0,0
        };

        
    // Constructor

        // Constructor from external data.
        LUsolver(double* externalData, int size);

        LUsolver(int size);
        
        // Disable default constructor 
        LUsolver()=delete;  

    // Destructor

        ~LUsolver();

    // Member function

        // Access

            //- (i, j) element access.
            inline double& operator()(const unsigned int i,const unsigned int j);

            //- (i, j) element access. 
            inline double operator() (const unsigned int i,const unsigned int j)const;
 
            //- pivotIndice access.
            inline unsigned int& pivotIndice(const unsigned int i);

            //- pivotIndice access.
            inline unsigned int pivotIndice(const unsigned int i) const;

inline __m256d load256d(const double* p)
{
    return _mm256_loadu_pd(p);
}

inline void store256d(double* p, __m256d v)
{
    _mm256_storeu_pd(p, v);
}

inline __m128d load128d(const double* p)
{
    return _mm_loadu_pd(p);
}

inline void store128d(double* p, __m128d v)
{
    _mm_storeu_pd(p, v);
}

        // Print the variable

            void printMatrix(double*A,int mRows, int nCol);

            void printMatrix();

            void printPivotIndice();

        // Block LU decomposition algorithm

            // using 4*4 block.
            void Block4LUDecompose();

            // Solve Ax=b.
            void xSolve_Serial(double* __restrict__ b);

            void xSolve_Vec_0(double* __restrict__ b);
            void xSolve_Vec_1(double* __restrict__ b);
            void xSolve_Vec_2(double* __restrict__ b);
            void xSolve_Vec_3(double* __restrict__ b);
            void xSolve(double* __restrict__ b);            

            // Sub functions

                // LU decomposition of 4*4 block.
                void LUDecompose4(unsigned int k0);
                //void LUDecompose4Old(unsigned int k0);
                //void LUDecompose4();
                void LUDecompose4_2();           
                void LUDecompose4_3();        
                
                //void forwardSitituate4(unsigned int k0,unsigned int k1);
                void forwardSitituate4_0(unsigned int k0,unsigned int k1);
                void forwardSitituate4_1(unsigned int k0,unsigned int k1);
                void forwardSitituate4_2(unsigned int k0,unsigned int k1);
                void forwardSitituate4_3(unsigned int k0,unsigned int k1);

                //void backSitituate4(unsigned int k0,unsigned int k1);
                void backSitituate4_0(unsigned int k0,unsigned int k1);
                void backSitituate4_1(unsigned int k0,unsigned int k1);
                void backSitituate4_2(unsigned int k0,unsigned int k1);
                void backSitituate4_3(unsigned int k0,unsigned int k1);                

                void permutation0(unsigned int k0,unsigned int k1);
                void permutation1(unsigned int k0,unsigned int k1);
                void permutation2(unsigned int k0,unsigned int k1);
                void permutation3(unsigned int k0,unsigned int k1);
                //void permutationold(unsigned int k0,unsigned int k1);
                //void permutation();
                void permutation_2();
                void permutation_3();

                //void UpdateL22U22(unsigned int k0,unsigned int k1);
                //void UpdateL22U22_Vec(unsigned int k0,unsigned int k1);
                //void UpdateL22U22_Vec2(unsigned int k0,unsigned int k1);   
                void UpdateL22U22_Vec2_0(unsigned int k0,unsigned int k1);   
                void UpdateL22U22_Vec2_1(unsigned int k0,unsigned int k1);  
                void UpdateL22U22_Vec2_2(unsigned int k0,unsigned int k1);  
                void UpdateL22U22_Vec2_3(unsigned int k0,unsigned int k1);   

                void UpdateL22U22_Vec_38_00(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);
                void UpdateL22U22_Vec_38_10(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);
                void UpdateL22U22_Vec_38_20(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);

                void UpdateL22U22_Vec_38_04(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);
                void UpdateL22U22_Vec_38_14(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);
                void UpdateL22U22_Vec_38_24(unsigned int k0,unsigned int k1, unsigned int rowTimes, unsigned int colTimes);

        // inverse matrix
        void calcInvMatrix(double* __restrict__ invMatrix);
        void setInvMatrix(double* __restrict__ invMatrix);
        void invMatrix56(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix52(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix48(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix44(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix40(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix36(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix32(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix28(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix24(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix20(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix16(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix12(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix8(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix4(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix3(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix2(double* __restrict__ invMatrix, const int m, const int n);
        void invMatrix1(double* __restrict__ invMatrix, const int m, const int n);

        // Reassign a square matrix
        void ReAssign(double* externalData);

        inline double hsum4(__m256d v);// result = v0+v1+v2+v3

        inline __m256d hsum4x4(__m256d sum0, __m256d sum1,__m256d sum2, __m256d sum3) const
        {
            __m256d t0 = _mm256_hadd_pd(sum0, sum1);  // [a0+a1, b0+b1, a2+a3, b2+b3]
            __m256d t1 = _mm256_hadd_pd(sum2, sum3);  // [c0+c1, d0+d1, c2+c3, d2+d3]
            __m256d t2 = _mm256_permute4x64_pd (t0, _MM_SHUFFLE(3,1,2,0)); // [a0+a1, a2+a3, b0+b1,  b2+b3]
            __m256d t3 = _mm256_permute4x64_pd(t1, _MM_SHUFFLE(3,1,2,0)); // [c0+c1, c2+c3, d0+d1, d2+d3]
            __m256d t4 = _mm256_hadd_pd(t2, t3); // [a0+a1+a2+a3, c0+c1+c2+c3, b0+b1+b2+b3, d0+d1+d2+d3]
            __m256d t5 = _mm256_permute4x64_pd(t4, _MM_SHUFFLE(3,1,2,0));
            return t5;
        }

        inline double get_elem0(__m256d vec) const
        {
            return _mm256_cvtsd_f64(vec); 
        }

        inline double get_elem1(__m256d vec) const
        {
            __m128d low = _mm256_castpd256_pd128(vec); 
            return _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
        }

        inline double get_elem2(__m256d vec) const
        {
            __m128d high = _mm256_extractf128_pd(vec, 1); 
            return _mm_cvtsd_f64(high);
        }

        inline __m128d get_high(__m256d vec) const
        {
            return _mm256_extractf128_pd(vec, 1);
        }

        inline __m128d get_low(__m256d vec) const
        {
            return _mm256_castpd256_pd128(vec);
        }

        inline double get_elem3(__m256d vec) const
        {
            __m128d high = _mm256_extractf128_pd(vec, 1); 
            return _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
        }

        inline void transpose4x4_pd(__m256d& v0,__m256d& v1,__m256d& v2,__m256d& v3);
};



} // End namespace Foam

#include "LUsolverI.H"



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
