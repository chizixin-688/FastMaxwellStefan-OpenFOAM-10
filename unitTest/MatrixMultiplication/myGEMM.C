

/*---------------------------------------------------------------------------*\
  Copyright (C) 2026 Zixin Chi
  All rights reserved.

  This file is an original work developed independently by Zixin Chi.

  This program is distributed under the terms of the
  GNU General Public License v3.0.

  Description
      This file is used to compute generalized Fick diffusion coefficient
      using binary diffusion coefficient. providing Matrix Multiplication
      function

  Author
      Zixin Chi <chizixin@buaa.edu.cn>

\*---------------------------------------------------------------------------*/


#include "myGEMM.h"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void MatrixMultiply484_0
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    for(int i = 0; i < N; i=i+4)
    {
        for(int j = 0; j < N; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
    }
}

void MatrixMultiply484_1
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{

    for(int i = 0; i < N-1; i=i+4)
    {
        for(int j = 0; j < N-1; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-1;

                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {
            int j = N-1;

            double Ci0j0 = 0;
            double Ci1j0 = 0;
            double Ci2j0 = 0;
            double Ci3j0 = 0;
            for(int k = 0; k < N-1; k=k+4)
            {
                double Bk0j0 = matB[(k+0)*N+(j+0)];
                double Ai0k0 = matA[(i+0)*N+(k+0)];
                Ci0j0 = Ai0k0*Bk0j0+Ci0j0;
                double Ai1k0 = matA[(i+1)*N+(k+0)];
                Ci1j0 = Ai1k0*Bk0j0+Ci1j0;
                double Ai2k0 = matA[(i+2)*N+(k+0)];
                Ci2j0 = Ai2k0*Bk0j0+Ci2j0;
                double Ai3k0 = matA[(i+3)*N+(k+0)];
                Ci3j0 = Ai3k0*Bk0j0+Ci3j0;
                double Bk1j0 = matB[(k+1)*N+(j+0)];
                double Ai0k1 = matA[(i+0)*N+(k+1)];
                Ci0j0 = Ai0k1*Bk1j0+Ci0j0;
                double Ai1k1 = matA[(i+1)*N+(k+1)];
                Ci1j0 = Ai1k1*Bk1j0+Ci1j0;
                double Ai2k1 = matA[(i+2)*N+(k+1)];
                Ci2j0 = Ai2k1*Bk1j0+Ci2j0;
                double Ai3k1 = matA[(i+3)*N+(k+1)];
                Ci3j0 = Ai3k1*Bk1j0+Ci3j0;
                double Bk2j0 = matB[(k+2)*N+j+0];
                double Ai0k2 = matA[(i+0)*N+(k+2)];
                Ci0j0 = Ai0k2*Bk2j0+Ci0j0;
                double Ai1k2 = matA[(i+1)*N+(k+2)];
                Ci1j0 = Ai1k2*Bk2j0+Ci1j0;
                double Ai2k2 = matA[(i+2)*N+(k+2)];
                Ci2j0 = Ai2k2*Bk2j0+Ci2j0;
                double Ai3k2 = matA[(i+3)*N+(k+2)];
                Ci3j0 = Ai3k2*Bk2j0+Ci3j0;
                double Bk3j0 = matB[(k+3)*N+j+0];
                double Ai0k3 = matA[(i+0)*N+(k+3)];
                Ci0j0 = Ai0k3*Bk3j0+Ci0j0;
                double Ai1k3v = matA[(i+1)*N+(k+3)];
                Ci1j0 = Ai1k3v*Bk3j0+Ci1j0;
                double Ai2k3v = matA[(i+2)*N+(k+3)];
                Ci2j0 = Ai2k3v*Bk3j0+Ci2j0;
                double Ai3k3v = matA[(i+3)*N+(k+3)];
                Ci3j0 = Ai3k3v*Bk3j0+Ci3j0;
            }
            {
                int k = N-1;
                double Bk0j0 = matB[(k+0)*N+(j+0)];
                double Ai0k0 = matA[(i+0)*N+(k+0)];
                Ci0j0 = Ai0k0*Bk0j0+Ci0j0;
                double Ai1k0 = matA[(i+1)*N+(k+0)];
                Ci1j0 = Ai1k0*Bk0j0+Ci1j0;
                double Ai2k0 = matA[(i+2)*N+(k+0)];
                Ci2j0 = Ai2k0*Bk0j0+Ci2j0;
                double Ai3k0 = matA[(i+3)*N+(k+0)];
                Ci3j0 = Ai3k0*Bk0j0+Ci3j0;
            }
            matC[(i+0)*(N)+j+0] = Ci0j0;
            matC[(i+1)*(N)+j+0] = Ci1j0;
            matC[(i+2)*(N)+j+0] = Ci2j0;
            matC[(i+3)*(N)+j+0] = Ci3j0;
        }
    }
    {
        int i = N-1;
        for(int j = 0; j < N-1; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci0j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14
                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);
                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);
                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);
                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);
                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);
                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);
                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);
            }
            {
                int k = N-1;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14
                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
        }
        {
            int j = N-1;
            double Ci0j0 = 0;
            for(int k = 0; k < N-1; k=k+4)
            {
                double Bk0j0 = matB[(k+0)*N+(j+0)];
                double Ai0k0 = matA[(i+0)*N+(k+0)];
                Ci0j0 = Ai0k0*Bk0j0+Ci0j0;
                double Bk1j0 = matB[(k+1)*N+(j+0)];
                double Ai0k1 = matA[(i+0)*N+(k+1)];
                Ci0j0 = Ai0k1*Bk1j0+Ci0j0;
                double Bk2j0 = matB[(k+2)*N+j+0];
                double Ai0k2 = matA[(i+0)*N+(k+2)];
                Ci0j0 = Ai0k2*Bk2j0+Ci0j0;
                double Bk3j0 = matB[(k+3)*N+j+0];
                double Ai0k3 = matA[(i+0)*N+(k+3)];
                Ci0j0 = Ai0k3*Bk3j0+Ci0j0;
            }
            {
                int k = N-1;
                double Bk0j0 = matB[(k+0)*N+(j+0)];
                double Ai0k0 = matA[(i+0)*N+(k+0)];
                Ci0j0 = Ai0k0*Bk0j0+Ci0j0;
            }
            matC[(i+0)*(N)+j+0] = Ci0j0;
        }
    }
}

void MatrixMultiply484_2
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    for(int i = 0; i < N-2; i=i+4)
    {
        for(int j = 0; j < N-2; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();
            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14
                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);
                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);
                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);
                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);
                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);
                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);
                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);
                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);
                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);
                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);
                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-2;

                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {// remain j
            int j = N-2;

            __m128d Ci0j01v = _mm_setzero_pd();
            __m128d Ci1j01v = _mm_setzero_pd();
            __m128d Ci2j01v = _mm_setzero_pd();
            __m128d Ci3j01v = _mm_setzero_pd();

            for(int k = 0; k < N-2; k=k+4)
            {
                __m128d Bk0j01 = _mm_loadu_pd(&matB[(k+0)*N+(j+0)]);
                
                __m128d Ai0k0 = _mm_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k0,Bk0j01,Ci0j01v);

                __m128d Ai1k0 = _mm_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k0,Bk0j01,Ci1j01v);

                __m128d Ai2k0 = _mm_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k0,Bk0j01,Ci2j01v);

                __m128d Ai3k0 = _mm_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k0,Bk0j01,Ci3j01v);


                __m128d Bk1j01 = _mm_loadu_pd(&matB[(k+1)*N+(j+0)]);

                __m128d Ai0k1 = _mm_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k1,Bk1j01,Ci0j01v);

                __m128d Ai1k1 = _mm_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k1,Bk1j01,Ci1j01v);

                __m128d Ai2k1 = _mm_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k1,Bk1j01,Ci2j01v);

                __m128d Ai3k1 = _mm_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k1,Bk1j01,Ci3j01v);


                __m128d Bk2j01 = _mm_loadu_pd(&matB[(k+2)*N+(j+0)]);

                __m128d Ai0k2 = _mm_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k2,Bk2j01,Ci0j01v);

                __m128d Ai1k2 = _mm_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k2,Bk2j01,Ci1j01v);

                __m128d Ai2k2 = _mm_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k2,Bk2j01,Ci2j01v);

                __m128d Ai3k2 = _mm_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k2,Bk2j01,Ci3j01v);


                __m128d Bk3j01 = _mm_loadu_pd(&matB[(k+3)*N+(j+0)]);

                __m128d Ai0k3 = _mm_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k3,Bk3j01,Ci0j01v);

                __m128d Ai1k3 = _mm_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k3,Bk3j01,Ci1j01v);

                __m128d Ai2k3 = _mm_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k3,Bk3j01,Ci2j01v);

                __m128d Ai3k3 = _mm_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k3,Bk3j01,Ci3j01v);

            }
            {
                int k = N-2;

                __m128d Bk0j01 = _mm_loadu_pd(&matB[(k+0)*N+(j+0)]);
                
                __m128d Ai0k0 = _mm_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k0,Bk0j01,Ci0j01v);

                __m128d Ai1k0 = _mm_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k0,Bk0j01,Ci1j01v);

                __m128d Ai2k0 = _mm_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k0,Bk0j01,Ci2j01v);

                __m128d Ai3k0 = _mm_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k0,Bk0j01,Ci3j01v);


                __m128d Bk1j01 = _mm_loadu_pd(&matB[(k+1)*N+(j+0)]);

                __m128d Ai0k1 = _mm_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j01v = _mm_fmadd_pd(Ai0k1,Bk1j01,Ci0j01v);

                __m128d Ai1k1 = _mm_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j01v = _mm_fmadd_pd(Ai1k1,Bk1j01,Ci1j01v);

                __m128d Ai2k1 = _mm_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j01v = _mm_fmadd_pd(Ai2k1,Bk1j01,Ci2j01v);

                __m128d Ai3k1 = _mm_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j01v = _mm_fmadd_pd(Ai3k1,Bk1j01,Ci3j01v);
            }

            _mm_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j01v);
            _mm_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j01v);
            _mm_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j01v);
            _mm_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j01v);
        }
    }
    {
        int i = N-2;
        for(int j = 0; j < N-2; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);
            }
            {
                int k = N-2;

                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
        }
        {// remain j
            int j = N-2;

            __m128d Ci0j01 = _mm_setzero_pd();
            __m128d Ci1j01 = _mm_setzero_pd();

            for(int k = 0; k < N-2; k=k+4)
            {
                __m128d Bk0j01 = _mm_loadu_pd(&matB[(k+0)*N+(j+0)]);
                
                __m128d Ai0k0 = _mm_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k0,Bk0j01,Ci0j01);

                __m128d Ai1k0 = _mm_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k0,Bk0j01,Ci1j01);



                __m128d Bk1j01 = _mm_loadu_pd(&matB[(k+1)*N+(j+0)]);

                __m128d Ai0k1 = _mm_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k1,Bk1j01,Ci0j01);

                __m128d Ai1k1 = _mm_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k1,Bk1j01,Ci1j01);


                __m128d Bk2j01 = _mm_loadu_pd(&matB[(k+2)*N+(j+0)]);

                __m128d Ai0k2 = _mm_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k2,Bk2j01,Ci0j01);

                __m128d Ai1k2 = _mm_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k2,Bk2j01,Ci1j01);


                __m128d Bk3j01 = _mm_loadu_pd(&matB[(k+3)*N+(j+0)]);

                __m128d Ai0k3 = _mm_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k3,Bk3j01,Ci0j01);

                __m128d Ai1k3 = _mm_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k3,Bk3j01,Ci1j01);

            }
            {
                int k = N-2;

                __m128d Bk0j01 = _mm_loadu_pd(&matB[(k+0)*N+(j+0)]);
                
                __m128d Ai0k0 = _mm_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k0,Bk0j01,Ci0j01);

                __m128d Ai1k0 = _mm_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k0,Bk0j01,Ci1j01);


                __m128d Bk1j01 = _mm_loadu_pd(&matB[(k+1)*N+(j+0)]);

                __m128d Ai0k1 = _mm_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j01 = _mm_fmadd_pd(Ai0k1,Bk1j01,Ci0j01);

                __m128d Ai1k1 = _mm_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j01 = _mm_fmadd_pd(Ai1k1,Bk1j01,Ci1j01);
            }

            _mm_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j01);
            _mm_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j01);
        }
    }
}

void MatrixMultiply484_3
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    //int remain4 = N%4;

    for(int i = 0; i < N-3; i=i+4)
    {
        for(int j = 0; j < N-3; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-3;

                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {// remain j
            int j = N-3;

            __m256d Ci0j02v = _mm256_setzero_pd();
            __m256d Ci1j02v = _mm256_setzero_pd();
            __m256d Ci2j02v = _mm256_setzero_pd();
            __m256d Ci3j02v = _mm256_setzero_pd();

            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j02v = _mm256_setr_pd(matB[(k+0)*N+(j+0)],matB[(k+0)*N+(j+1)],matB[(k+0)*N+(j+2)],0);
                
                __m256d Ai0k0 = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k0,Bk0j02v,Ci0j02v);

                __m256d Ai1k0 = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k0,Bk0j02v,Ci1j02v);

                __m256d Ai2k0 = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k0,Bk0j02v,Ci2j02v);

                __m256d Ai3k0 = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k0,Bk0j02v,Ci3j02v);


                __m256d Bk1j02v = _mm256_setr_pd(matB[(k+1)*N+(j+0)],matB[(k+1)*N+(j+1)],matB[(k+1)*N+(j+2)],0);

                __m256d Ai0k1 = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k1,Bk1j02v,Ci0j02v);

                __m256d Ai1k1 = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k1,Bk1j02v,Ci1j02v);

                __m256d Ai2k1 = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k1,Bk1j02v,Ci2j02v);

                __m256d Ai3k1 = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k1,Bk1j02v,Ci3j02v);


                __m256d Bk2j02v = _mm256_setr_pd(matB[(k+2)*N+(j+0)],matB[(k+2)*N+(j+1)],matB[(k+2)*N+(j+2)],0);

                __m256d Ai0k2 = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k2,Bk2j02v,Ci0j02v);

                __m256d Ai1k2 = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k2,Bk2j02v,Ci1j02v);

                __m256d Ai2k2 = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k2,Bk2j02v,Ci2j02v);

                __m256d Ai3k2 = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k2,Bk2j02v,Ci3j02v);


                __m256d Bk3j02v = _mm256_setr_pd(matB[(k+3)*N+(j+0)],matB[(k+3)*N+(j+1)],matB[(k+3)*N+(j+2)],0);

                __m256d Ai0k3 = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k3,Bk3j02v,Ci0j02v);

                __m256d Ai1k3 = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k3,Bk3j02v,Ci1j02v);

                __m256d Ai2k3 = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k3,Bk3j02v,Ci2j02v);

                __m256d Ai3k3 = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k3,Bk3j02v,Ci3j02v);

            }
            {
                int k = N-3;

                __m256d Bk0j02v = _mm256_setr_pd(matB[(k+0)*N+(j+0)],matB[(k+0)*N+(j+1)],matB[(k+0)*N+(j+2)],0);
                
                __m256d Ai0k0 = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k0,Bk0j02v,Ci0j02v);

                __m256d Ai1k0 = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k0,Bk0j02v,Ci1j02v);

                __m256d Ai2k0 = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k0,Bk0j02v,Ci2j02v);

                __m256d Ai3k0 = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k0,Bk0j02v,Ci3j02v);


                __m256d Bk1j02v = _mm256_setr_pd(matB[(k+1)*N+(j+0)],matB[(k+1)*N+(j+1)],matB[(k+1)*N+(j+2)],0);

                __m256d Ai0k1 = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k1,Bk1j02v,Ci0j02v);

                __m256d Ai1k1 = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k1,Bk1j02v,Ci1j02v);

                __m256d Ai2k1 = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k1,Bk1j02v,Ci2j02v);

                __m256d Ai3k1 = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k1,Bk1j02v,Ci3j02v);


                __m256d Bk2j02v = _mm256_setr_pd(matB[(k+2)*N+(j+0)],matB[(k+2)*N+(j+1)],matB[(k+2)*N+(j+2)],0);

                __m256d Ai0k2 = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k2,Bk2j02v,Ci0j02v);

                __m256d Ai1k2 = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k2,Bk2j02v,Ci1j02v);

                __m256d Ai2k2 = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k2,Bk2j02v,Ci2j02v);

                __m256d Ai3k2 = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j02v = _mm256_fmadd_pd(Ai3k2,Bk2j02v,Ci3j02v);
            }
            matC[(i+0)*(N)+j+0] = get_elem0(Ci0j02v);
            matC[(i+0)*(N)+j+1] = get_elem1(Ci0j02v);
            matC[(i+0)*(N)+j+2] = get_elem2(Ci0j02v);

            matC[(i+1)*(N)+j+0] = get_elem0(Ci1j02v);
            matC[(i+1)*(N)+j+1] = get_elem1(Ci1j02v);
            matC[(i+1)*(N)+j+2] = get_elem2(Ci1j02v);

            matC[(i+2)*(N)+j+0] = get_elem0(Ci2j02v);
            matC[(i+2)*(N)+j+1] = get_elem1(Ci2j02v);
            matC[(i+2)*(N)+j+2] = get_elem2(Ci2j02v);

            matC[(i+3)*(N)+j+0] = get_elem0(Ci3j02v);
            matC[(i+3)*(N)+j+1] = get_elem1(Ci3j02v);
            matC[(i+3)*(N)+j+2] = get_elem2(Ci3j02v);

            //_mm256_storeu_pd(&C[(i+0)*(N)+j+0],Ci0j02v);
            //_mm256_storeu_pd(&C[(i+1)*(N)+j+0],Ci1j02v);
            //_mm256_storeu_pd(&C[(i+2)*(N)+j+0],Ci2j02v);
            //_mm256_storeu_pd(&C[(i+3)*(N)+j+0],Ci3j02v);
        }
    }
    {
        int i = N-3;
        for(int j = 0; j < N-3; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);
            }
            {
                int k = N-3;

                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
        }
        {// remain j
            int j = N-3;

            __m256d Ci0j02v = _mm256_setzero_pd();
            __m256d Ci1j02v = _mm256_setzero_pd();
            __m256d Ci2j02v = _mm256_setzero_pd();

            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j02v = _mm256_setr_pd(matB[(k+0)*N+(j+0)],matB[(k+0)*N+(j+1)],matB[(k+0)*N+(j+2)],0);
                
                __m256d Ai0k0 = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k0,Bk0j02v,Ci0j02v);

                __m256d Ai1k0 = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k0,Bk0j02v,Ci1j02v);

                __m256d Ai2k0 = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k0,Bk0j02v,Ci2j02v);

                __m256d Bk1j02v = _mm256_setr_pd(matB[(k+1)*N+(j+0)],matB[(k+1)*N+(j+1)],matB[(k+1)*N+(j+2)],0);

                __m256d Ai0k1 = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k1,Bk1j02v,Ci0j02v);

                __m256d Ai1k1 = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k1,Bk1j02v,Ci1j02v);

                __m256d Ai2k1 = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k1,Bk1j02v,Ci2j02v);

                __m256d Bk2j02v = _mm256_setr_pd(matB[(k+2)*N+(j+0)],matB[(k+2)*N+(j+1)],matB[(k+2)*N+(j+2)],0);

                __m256d Ai0k2 = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k2,Bk2j02v,Ci0j02v);

                __m256d Ai1k2 = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k2,Bk2j02v,Ci1j02v);

                __m256d Ai2k2 = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k2,Bk2j02v,Ci2j02v);

                __m256d Bk3j02v = _mm256_setr_pd(matB[(k+3)*N+(j+0)],matB[(k+3)*N+(j+1)],matB[(k+3)*N+(j+2)],0);

                __m256d Ai0k3 = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k3,Bk3j02v,Ci0j02v);

                __m256d Ai1k3 = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k3,Bk3j02v,Ci1j02v);

                __m256d Ai2k3 = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k3,Bk3j02v,Ci2j02v);
            }
            {
                int k = N-3;

                __m256d Bk0j02v = _mm256_setr_pd(matB[(k+0)*N+(j+0)],matB[(k+0)*N+(j+1)],matB[(k+0)*N+(j+2)],0);
                
                __m256d Ai0k0 = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k0,Bk0j02v,Ci0j02v);

                __m256d Ai1k0 = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k0,Bk0j02v,Ci1j02v);

                __m256d Ai2k0 = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k0,Bk0j02v,Ci2j02v);


                __m256d Bk1j02v = _mm256_setr_pd(matB[(k+1)*N+(j+0)],matB[(k+1)*N+(j+1)],matB[(k+1)*N+(j+2)],0);

                __m256d Ai0k1 = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k1,Bk1j02v,Ci0j02v);

                __m256d Ai1k1 = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k1,Bk1j02v,Ci1j02v);

                __m256d Ai2k1 = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k1,Bk1j02v,Ci2j02v);


                __m256d Bk2j02v = _mm256_setr_pd(matB[(k+2)*N+(j+0)],matB[(k+2)*N+(j+1)],matB[(k+2)*N+(j+2)],0);

                __m256d Ai0k2 = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j02v = _mm256_fmadd_pd(Ai0k2,Bk2j02v,Ci0j02v);

                __m256d Ai1k2 = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j02v = _mm256_fmadd_pd(Ai1k2,Bk2j02v,Ci1j02v);

                __m256d Ai2k2 = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j02v = _mm256_fmadd_pd(Ai2k2,Bk2j02v,Ci2j02v);
            }

            matC[(i+0)*(N)+j+0] = get_elem0(Ci0j02v);
            matC[(i+0)*(N)+j+1] = get_elem1(Ci0j02v);
            matC[(i+0)*(N)+j+2] = get_elem2(Ci0j02v);

            matC[(i+1)*(N)+j+0] = get_elem0(Ci1j02v);
            matC[(i+1)*(N)+j+1] = get_elem1(Ci1j02v);
            matC[(i+1)*(N)+j+2] = get_elem2(Ci1j02v);

            matC[(i+2)*(N)+j+0] = get_elem0(Ci2j02v);
            matC[(i+2)*(N)+j+1] = get_elem1(Ci2j02v);
            matC[(i+2)*(N)+j+2] = get_elem2(Ci2j02v);
        }
    }
}


void MatrixMultiply484_4
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    //int remain4 = N%4;

    for(int i = 0; i < N; i=i+4)
    {
        for(int j = 0; j < N-4; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {
            int j = N-4;

            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            for(int k = 0; k < N; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
            }
            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);
        }
    }
}


void MatrixMultiply484_5
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    //int remain4 = N%4;

    for(int i = 0; i < N-1; i=i+4)
    {
        for(int j = 0; j < N-5; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-1;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {
            int j = N-5;
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
            }
            {
                int k = N-1;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);
        }
        {
            int j = N-1;
            double Ci0j0 = 0;
            double Ci1j0 = 0;
            double Ci2j0 = 0;
            double Ci3j0 = 0;

            for(int k = 0; k < N-1; k=k+4)
            {
                double Bk0j0 = (matB[(k+0)*N+j+0]);//13
                double Ai0k0 = (matA[(i+0)*N+(k+0)]);//15
                Ci0j0 = Ci0j0 + Ai0k0*Bk0j0;

                double Ai1k0 = (matA[(i+1)*N+(k+0)]);
                Ci1j0 = Ci1j0 + Ai1k0*Bk0j0;

                double Ai2k0 = (matA[(i+2)*N+(k+0)]);
                Ci2j0 = Ci2j0 + Ai2k0*Bk0j0;

                double Ai3k0 = (matA[(i+3)*N+(k+0)]);
                Ci3j0 = Ci3j0 + Ai3k0*Bk0j0;


                double Bk1j0 = (matB[(k+1)*N+j+0]);

                double Ai0k1 = (matA[(i+0)*N+(k+1)]);
                Ci0j0 = Ci0j0 + Ai0k1*Bk1j0;

                double Ai1k1 = (matA[(i+1)*N+(k+1)]);
                Ci1j0 = Ci1j0 + Ai1k1*Bk1j0;

                double Ai2k1 = (matA[(i+2)*N+(k+1)]);
                Ci2j0 = Ci2j0 + Ai2k1*Bk1j0;

                double Ai3k1 = (matA[(i+3)*N+(k+1)]);
                Ci3j0 = Ci3j0 + Ai3k1*Bk1j0;


                double Bk2j03v = (matB[(k+2)*N+j+0]);

                double Ai0k2v = (matA[(i+0)*N+(k+2)]);
                Ci0j0 = Ci0j0 + Ai0k2v*Bk2j03v;

                double Ai1k2v = (matA[(i+1)*N+(k+2)]);
                Ci1j0 = Ci1j0 + Ai1k2v*Bk2j03v;

                double Ai2k2v = (matA[(i+2)*N+(k+2)]);
                Ci2j0 = Ci2j0 + Ai2k2v*Bk2j03v;

                double Ai3k2v = (matA[(i+3)*N+(k+2)]);
                Ci3j0 = Ci3j0 + Ai3k2v*Bk2j03v;


                double Bk3j03v = (matB[(k+3)*N+j+0]);

                double Ai0k3v = (matA[(i+0)*N+(k+3)]);
                Ci0j0 = Ci0j0 + Ai0k3v*Bk3j03v;

                double Ai1k3v = (matA[(i+1)*N+(k+3)]);
                Ci1j0 = Ci1j0 + Ai1k3v*Bk3j03v;

                double Ai2k3v = (matA[(i+2)*N+(k+3)]);
                Ci2j0 = Ci2j0 + Ai2k3v*Bk3j03v;

                double Ai3k3v = (matA[(i+3)*N+(k+3)]);
                Ci3j0 = Ci3j0 + Ai3k3v*Bk3j03v;
            }
            {
                int k = N-1;
                double Bk0j0 = (matB[(k+0)*N+j+0]);//13

                double Ai0k0 = (matA[(i+0)*N+(k+0)]);//15
                Ci0j0 = Ci0j0 + Ai0k0*Bk0j0;

                double Ai1k0 = (matA[(i+1)*N+(k+0)]);
                Ci1j0 = Ci1j0 + Ai1k0*Bk0j0;

                double Ai2k0 = (matA[(i+2)*N+(k+0)]);
                Ci2j0 = Ci2j0 + Ai2k0*Bk0j0;

                double Ai3k0 = (matA[(i+3)*N+(k+0)]);
                Ci3j0 = Ci3j0 + Ai3k0*Bk0j0;
            }

            matC[(i+0)*(N)+j+0] = Ci0j0;
            matC[(i+1)*(N)+j+0] = Ci1j0;
            matC[(i+2)*(N)+j+0] = Ci2j0;
            matC[(i+3)*(N)+j+0] = Ci3j0;
        }
    }
    {
        int i = N-1;

        for(int j = 0; j < N-5; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);



                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);



                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

            }
            {
                int k = N-1;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
        }
        {
            int j = N-5;
            __m256d Ci0j03v = _mm256_setzero_pd();

            for(int k = 0; k < N-1; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
            }
            {
                int k = N-1;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
        }
        {
            int j = N-1;
            double Ci0j0 = 0;


            for(int k = 0; k < N-1; k=k+4)
            {
                double Bk0j0 = (matB[(k+0)*N+j+0]);//13
                double Ai0k0 = (matA[(i+0)*N+(k+0)]);//15
                Ci0j0 = Ci0j0 + Ai0k0*Bk0j0;



                double Bk1j0 = (matB[(k+1)*N+j+0]);

                double Ai0k1 = (matA[(i+0)*N+(k+1)]);
                Ci0j0 = Ci0j0 + Ai0k1*Bk1j0;


                double Bk2j03v = (matB[(k+2)*N+j+0]);

                double Ai0k2v = (matA[(i+0)*N+(k+2)]);
                Ci0j0 = Ci0j0 + Ai0k2v*Bk2j03v;



                double Bk3j03v = (matB[(k+3)*N+j+0]);

                double Ai0k3v = (matA[(i+0)*N+(k+3)]);
                Ci0j0 = Ci0j0 + Ai0k3v*Bk3j03v;
            }
            {
                int k = N-1;
                double Bk0j0 = (matB[(k+0)*N+j+0]);//13

                double Ai0k0 = (matA[(i+0)*N+(k+0)]);//15
                Ci0j0 = Ci0j0 + Ai0k0*Bk0j0;
            }

            matC[(i+0)*(N)+j+0] = Ci0j0;
        }
    }
}

void MatrixMultiply484_6
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    //int remain4 = N%4;

    for(int i = 0; i < N-2; i=i+4)
    {
        for(int j = 0; j < N-6; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-2;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {
            int j = N-6;
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m128d Ci0j45v = _mm_setzero_pd();
            __m128d Ci1j45v = _mm_setzero_pd();
            __m128d Ci2j45v = _mm_setzero_pd();
            __m128d Ci3j45v = _mm_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m128d Bk0j45v = _mm_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k0v),Bk0j45v,Ci0j45v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k0v),Bk0j45v,Ci1j45v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k0v),Bk0j45v,Ci2j45v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k0v),Bk0j45v,Ci3j45v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m128d Bk1j45v = _mm_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k1v),Bk1j45v,Ci0j45v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k1v),Bk1j45v,Ci1j45v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k1v),Bk1j45v,Ci2j45v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k1v),Bk1j45v,Ci3j45v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m128d Bk2j45v = _mm_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k2v),Bk2j45v,Ci0j45v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k2v),Bk2j45v,Ci1j45v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k2v),Bk2j45v,Ci2j45v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k2v),Bk2j45v,Ci3j45v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m128d Bk3j45v = _mm_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k3v),Bk3j45v,Ci0j45v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k3v),Bk3j45v,Ci1j45v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k3v),Bk3j45v,Ci2j45v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k3v),Bk3j45v,Ci3j45v);
            }
            {
                int k = N-2;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m128d Bk0j45v = _mm_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k0v),Bk0j45v,Ci0j45v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k0v),Bk0j45v,Ci1j45v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k0v),Bk0j45v,Ci2j45v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k0v),Bk0j45v,Ci3j45v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m128d Bk1j45v = _mm_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k1v),Bk1j45v,Ci0j45v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k1v),Bk1j45v,Ci1j45v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai2k1v),Bk1j45v,Ci2j45v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai3k1v),Bk1j45v,Ci3j45v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j45v);
            _mm_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j45v);
            _mm_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j45v);
            _mm_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j45v);
        }
    }
    {
        int i = N-2;
        for(int j = 0; j < N-6; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);
            }
            {
                int k = N-2;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
        }
        {
            int j = N-6;
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();

            __m128d Ci0j45v = _mm_setzero_pd();
            __m128d Ci1j45v = _mm_setzero_pd();
            for(int k = 0; k < N-2; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m128d Bk0j45v = _mm_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k0v),Bk0j45v,Ci0j45v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k0v),Bk0j45v,Ci1j45v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m128d Bk1j45v = _mm_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k1v),Bk1j45v,Ci0j45v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k1v),Bk1j45v,Ci1j45v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m128d Bk2j45v = _mm_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k2v),Bk2j45v,Ci0j45v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k2v),Bk2j45v,Ci1j45v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m128d Bk3j45v = _mm_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k3v),Bk3j45v,Ci0j45v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k3v),Bk3j45v,Ci1j45v);
            }
            {
                int k = N-2;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m128d Bk0j45v = _mm_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k0v),Bk0j45v,Ci0j45v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k0v),Bk0j45v,Ci1j45v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m128d Bk1j45v = _mm_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai0k1v),Bk1j45v,Ci0j45v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j45v = _mm_fmadd_pd(_mm256_castpd256_pd128(Ai1k1v),Bk1j45v,Ci1j45v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);

            _mm_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j45v);
            _mm_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j45v);
        }
    }
}

void MatrixMultiply484_7
(
    double* __restrict__ matC,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    int N
)
{
    //int remain4 = N%4;

    for(int i = 0; i < N-3; i=i+4)
    {
        for(int j = 0; j < N-7; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            __m256d Ci3j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k3v,Bk3j47v,Ci3j47v);
            }
            {
                int k = N-3;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k0v,Bk0j47v,Ci3j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k1v,Bk1j47v,Ci3j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j47v = _mm256_fmadd_pd(Ai3k2v,Bk2j47v,Ci3j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+4],Ci3j47v);
        }
        {
            int j = N-7;
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();
            __m256d Ci3j03v = _mm256_setzero_pd();

            __m256d Ci0j46v = _mm256_setzero_pd();
            __m256d Ci1j46v = _mm256_setzero_pd();
            __m256d Ci2j46v = _mm256_setzero_pd();
            __m256d Ci3j46v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j46v = _mm256_setr_pd(matB[(k+0)*N+j+4],matB[(k+0)*N+j+5],matB[(k+0)*N+j+6],0);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k0v,Bk0j46v,Ci0j46v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k0v,Bk0j46v,Ci1j46v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k0v,Bk0j46v,Ci2j46v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k0v,Bk0j46v,Ci3j46v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j46v = _mm256_setr_pd(matB[(k+1)*N+j+4],matB[(k+1)*N+j+5],matB[(k+1)*N+j+6],0);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k1v,Bk1j46v,Ci0j46v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k1v,Bk1j46v,Ci1j46v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k1v,Bk1j46v,Ci2j46v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k1v,Bk1j46v,Ci3j46v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j46v = _mm256_setr_pd(matB[(k+2)*N+j+4],matB[(k+2)*N+j+5],matB[(k+2)*N+j+6],0);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k2v,Bk2j46v,Ci0j46v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k2v,Bk2j46v,Ci1j46v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k2v,Bk2j46v,Ci2j46v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k2v,Bk2j46v,Ci3j46v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j46v = _mm256_setr_pd(matB[(k+3)*N+j+4],matB[(k+3)*N+j+5],matB[(k+3)*N+j+6],0);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k3v,Bk3j46v,Ci0j46v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k3v,Bk3j46v,Ci1j46v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k3v,Bk3j46v,Ci2j46v);

                __m256d Ai3k3v = _mm256_set1_pd(matA[(i+3)*N+(k+3)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k3v,Bk3j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k3v,Bk3j46v,Ci3j46v);
            }
            {
                int k = N-3;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j46v = _mm256_setr_pd(matB[(k+0)*N+j+4],matB[(k+0)*N+j+5],matB[(k+0)*N+j+6],0);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k0v,Bk0j46v,Ci0j46v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k0v,Bk0j46v,Ci1j46v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k0v,Bk0j46v,Ci2j46v);

                __m256d Ai3k0v = _mm256_set1_pd(matA[(i+3)*N+(k+0)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k0v,Bk0j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k0v,Bk0j46v,Ci3j46v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j46v = _mm256_setr_pd(matB[(k+1)*N+j+4],matB[(k+1)*N+j+5],matB[(k+1)*N+j+6],0);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k1v,Bk1j46v,Ci0j46v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k1v,Bk1j46v,Ci1j46v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k1v,Bk1j46v,Ci2j46v);

                __m256d Ai3k1v = _mm256_set1_pd(matA[(i+3)*N+(k+1)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k1v,Bk1j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k1v,Bk1j46v,Ci3j46v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j46v = _mm256_setr_pd(matB[(k+2)*N+j+4],matB[(k+2)*N+j+5],matB[(k+2)*N+j+6],0);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k2v,Bk2j46v,Ci0j46v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k2v,Bk2j46v,Ci1j46v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k2v,Bk2j46v,Ci2j46v);

                __m256d Ai3k2v = _mm256_set1_pd(matA[(i+3)*N+(k+2)]);
                Ci3j03v = _mm256_fmadd_pd(Ai3k2v,Bk2j03v,Ci3j03v);
                Ci3j46v = _mm256_fmadd_pd(Ai3k2v,Bk2j46v,Ci3j46v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);
            _mm256_storeu_pd(&matC[(i+3)*(N)+j+0],Ci3j03v);


            matC[(i+0)*(N)+j+4] = get_elem0(Ci0j46v);
            matC[(i+0)*(N)+j+5] = get_elem1(Ci0j46v);
            matC[(i+0)*(N)+j+6] = get_elem2(Ci0j46v);

            matC[(i+1)*(N)+j+4] = get_elem0(Ci1j46v);
            matC[(i+1)*(N)+j+5] = get_elem1(Ci1j46v);
            matC[(i+1)*(N)+j+6] = get_elem2(Ci1j46v);

            matC[(i+2)*(N)+j+4] = get_elem0(Ci2j46v);
            matC[(i+2)*(N)+j+5] = get_elem1(Ci2j46v);
            matC[(i+2)*(N)+j+6] = get_elem2(Ci2j46v);

            matC[(i+3)*(N)+j+4] = get_elem0(Ci3j46v);
            matC[(i+3)*(N)+j+5] = get_elem1(Ci3j46v);
            matC[(i+3)*(N)+j+6] = get_elem2(Ci3j46v);
        }
    }
    {
        int i = N-3;
        for(int j = 0; j < N-7; j=j+8)
        {
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();

            __m256d Ci0j47v = _mm256_setzero_pd();
            __m256d Ci1j47v = _mm256_setzero_pd();
            __m256d Ci2j47v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j47v = _mm256_loadu_pd(&matB[(k+3)*N+j+4]);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k3v,Bk3j47v,Ci0j47v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k3v,Bk3j47v,Ci1j47v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k3v,Bk3j47v,Ci2j47v);
            }
            {
                int k = N-3;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j47v = _mm256_loadu_pd(&matB[(k+0)*N+j+4]);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k0v,Bk0j47v,Ci0j47v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k0v,Bk0j47v,Ci1j47v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k0v,Bk0j47v,Ci2j47v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j47v = _mm256_loadu_pd(&matB[(k+1)*N+j+4]);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k1v,Bk1j47v,Ci0j47v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k1v,Bk1j47v,Ci1j47v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k1v,Bk1j47v,Ci2j47v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j47v = _mm256_loadu_pd(&matB[(k+2)*N+j+4]);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j47v = _mm256_fmadd_pd(Ai0k2v,Bk2j47v,Ci0j47v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j47v = _mm256_fmadd_pd(Ai1k2v,Bk2j47v,Ci1j47v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j47v = _mm256_fmadd_pd(Ai2k2v,Bk2j47v,Ci2j47v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+4],Ci0j47v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+4],Ci1j47v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+4],Ci2j47v);
        }
        {
            int j = N-7;
            __m256d Ci0j03v = _mm256_setzero_pd();
            __m256d Ci1j03v = _mm256_setzero_pd();
            __m256d Ci2j03v = _mm256_setzero_pd();

            __m256d Ci0j46v = _mm256_setzero_pd();
            __m256d Ci1j46v = _mm256_setzero_pd();
            __m256d Ci2j46v = _mm256_setzero_pd();
            for(int k = 0; k < N-3; k=k+4)
            {
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j46v = _mm256_setr_pd(matB[(k+0)*N+j+4],matB[(k+0)*N+j+5],matB[(k+0)*N+j+6],0);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k0v,Bk0j46v,Ci0j46v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k0v,Bk0j46v,Ci1j46v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k0v,Bk0j46v,Ci2j46v);


                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j46v = _mm256_setr_pd(matB[(k+1)*N+j+4],matB[(k+1)*N+j+5],matB[(k+1)*N+j+6],0);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k1v,Bk1j46v,Ci0j46v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k1v,Bk1j46v,Ci1j46v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k1v,Bk1j46v,Ci2j46v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j46v = _mm256_setr_pd(matB[(k+2)*N+j+4],matB[(k+2)*N+j+5],matB[(k+2)*N+j+6],0);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k2v,Bk2j46v,Ci0j46v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k2v,Bk2j46v,Ci1j46v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k2v,Bk2j46v,Ci2j46v);


                __m256d Bk3j03v = _mm256_loadu_pd(&matB[(k+3)*N+j+0]);
                __m256d Bk3j46v = _mm256_setr_pd(matB[(k+3)*N+j+4],matB[(k+3)*N+j+5],matB[(k+3)*N+j+6],0);

                __m256d Ai0k3v = _mm256_set1_pd(matA[(i+0)*N+(k+3)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k3v,Bk3j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k3v,Bk3j46v,Ci0j46v);

                __m256d Ai1k3v = _mm256_set1_pd(matA[(i+1)*N+(k+3)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k3v,Bk3j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k3v,Bk3j46v,Ci1j46v);

                __m256d Ai2k3v = _mm256_set1_pd(matA[(i+2)*N+(k+3)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k3v,Bk3j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k3v,Bk3j46v,Ci2j46v);
            }
            {
                int k = N-3;
                __m256d Bk0j03v = _mm256_loadu_pd(&matB[(k+0)*N+j+0]);//13
                __m256d Bk0j46v = _mm256_setr_pd(matB[(k+0)*N+j+4],matB[(k+0)*N+j+5],matB[(k+0)*N+j+6],0);//14

                __m256d Ai0k0v = _mm256_set1_pd(matA[(i+0)*N+(k+0)]);//15
                Ci0j03v = _mm256_fmadd_pd(Ai0k0v,Bk0j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k0v,Bk0j46v,Ci0j46v);

                __m256d Ai1k0v = _mm256_set1_pd(matA[(i+1)*N+(k+0)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k0v,Bk0j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k0v,Bk0j46v,Ci1j46v);

                __m256d Ai2k0v = _mm256_set1_pd(matA[(i+2)*N+(k+0)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k0v,Bk0j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k0v,Bk0j46v,Ci2j46v);

                __m256d Bk1j03v = _mm256_loadu_pd(&matB[(k+1)*N+j+0]);
                __m256d Bk1j46v = _mm256_setr_pd(matB[(k+1)*N+j+4],matB[(k+1)*N+j+5],matB[(k+1)*N+j+6],0);

                __m256d Ai0k1v = _mm256_set1_pd(matA[(i+0)*N+(k+1)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k1v,Bk1j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k1v,Bk1j46v,Ci0j46v);

                __m256d Ai1k1v = _mm256_set1_pd(matA[(i+1)*N+(k+1)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k1v,Bk1j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k1v,Bk1j46v,Ci1j46v);

                __m256d Ai2k1v = _mm256_set1_pd(matA[(i+2)*N+(k+1)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k1v,Bk1j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k1v,Bk1j46v,Ci2j46v);


                __m256d Bk2j03v = _mm256_loadu_pd(&matB[(k+2)*N+j+0]);
                __m256d Bk2j46v = _mm256_setr_pd(matB[(k+2)*N+j+4],matB[(k+2)*N+j+5],matB[(k+2)*N+j+6],0);

                __m256d Ai0k2v = _mm256_set1_pd(matA[(i+0)*N+(k+2)]);
                Ci0j03v = _mm256_fmadd_pd(Ai0k2v,Bk2j03v,Ci0j03v);
                Ci0j46v = _mm256_fmadd_pd(Ai0k2v,Bk2j46v,Ci0j46v);

                __m256d Ai1k2v = _mm256_set1_pd(matA[(i+1)*N+(k+2)]);
                Ci1j03v = _mm256_fmadd_pd(Ai1k2v,Bk2j03v,Ci1j03v);
                Ci1j46v = _mm256_fmadd_pd(Ai1k2v,Bk2j46v,Ci1j46v);

                __m256d Ai2k2v = _mm256_set1_pd(matA[(i+2)*N+(k+2)]);
                Ci2j03v = _mm256_fmadd_pd(Ai2k2v,Bk2j03v,Ci2j03v);
                Ci2j46v = _mm256_fmadd_pd(Ai2k2v,Bk2j46v,Ci2j46v);
            }

            _mm256_storeu_pd(&matC[(i+0)*(N)+j+0],Ci0j03v);
            _mm256_storeu_pd(&matC[(i+1)*(N)+j+0],Ci1j03v);
            _mm256_storeu_pd(&matC[(i+2)*(N)+j+0],Ci2j03v);


            matC[(i+0)*(N)+j+4] = get_elem0(Ci0j46v);
            matC[(i+0)*(N)+j+5] = get_elem1(Ci0j46v);
            matC[(i+0)*(N)+j+6] = get_elem2(Ci0j46v);

            matC[(i+1)*(N)+j+4] = get_elem0(Ci1j46v);
            matC[(i+1)*(N)+j+5] = get_elem1(Ci1j46v);
            matC[(i+1)*(N)+j+6] = get_elem2(Ci1j46v);

            matC[(i+2)*(N)+j+4] = get_elem0(Ci2j46v);
            matC[(i+2)*(N)+j+5] = get_elem1(Ci2j46v);
            matC[(i+2)*(N)+j+6] = get_elem2(Ci2j46v);
        }
    }
}


inline double get_elem0(__m256d vec) 
{
    return _mm256_cvtsd_f64(vec); 
}

inline double get_elem0(__m128d vec) 
{
    return _mm_cvtsd_f64(vec); 
}

inline double get_elem1(__m256d vec) 
{
    __m128d low = _mm256_castpd256_pd128(vec); 
    return _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
}

inline double get_elem1(__m128d vec) 
{
    return _mm_cvtsd_f64(_mm_unpackhi_pd(vec,vec)); 
}

inline double get_elem2(__m256d vec) 
{
    __m128d high = _mm256_extractf128_pd(vec, 1); 
    return _mm_cvtsd_f64(high);
}

inline double get_elem3(__m256d vec) 
{
    __m128d high = _mm256_extractf128_pd(vec, 1); 
    return _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
}

