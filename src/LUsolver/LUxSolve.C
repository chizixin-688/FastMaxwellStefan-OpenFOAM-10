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

//=============================================================================//

//---------------------------------
// 1. Standard C++ library headers
//---------------------------------
#include <iostream>
#include <cmath>

//---------------------------------
// 2. FastChemistry headers
//---------------------------------
#include "LUsolver.H"

//---------------------------------
// 3. SIMD / AVX2 headers
//---------------------------------
#include <immintrin.h>

//=============================================================================//
namespace Foam
{



void LUsolver::setInvMatrix
(
    double* __restrict__ invMatrix
)
{
    for(unsigned int i=0; i<this->N; i++)
    {
        this->invMatRowPtr[i] = &invMatrix[i*this->alignN];
        invMatrix[i*this->alignN+i] = 1.0;
    }


    for(unsigned int j = 0; j<this->N;j++)
    {
        if(this->pivotIndice_[j]==j)
        {
            continue;
        }
        unsigned int jTarget = this->pivotIndice_[j];
        std::swap(invMatRowPtr[jTarget], invMatRowPtr[j]);
    }


    
    /*int remain = this->N%4;
    int align = this->N - remain;
    std::vector<int> coins = {56,52,48,44,40,36,32,28,24,20,16,12,8,4};
    
    for (int i = 0; i < 14; i++) 
    { 
        int n = align / coins[i]; 
        this->callTable[i] = n; 
        align = align - n*coins[i]; 
    }*/

}

void LUsolver::calcInvMatrix
(
    double* __restrict__ invMatrix
)
{
    int remain = this->N%4;
    //this->setInvMatrix(invMatrix);
    int begin = 0;

    if(this->callTable[0]>0)
    {
        int times = this->callTable[0];
        (this->*invMatFuncTable[0])(invMatrix,begin,times);
        begin = begin + times*56;
    }

    if(this->callTable[1]>0)
    {
        int times = this->callTable[1];
        (this->*invMatFuncTable[1])(invMatrix,begin,times);
        begin = begin + times*52;
    }

    if(this->callTable[2]>0)
    {
        int times = this->callTable[2];
        (this->*invMatFuncTable[2])(invMatrix,begin,times);
        begin = begin + times*48;
    }

    if(this->callTable[3]>0)
    {
        int times = this->callTable[3];
        (this->*invMatFuncTable[3])(invMatrix,begin,times);
        begin = begin + times*44;
    }

    if(this->callTable[4]>0)
    {
        int times = this->callTable[4];
        (this->*invMatFuncTable[4])(invMatrix,begin,times);
        begin = begin + times*40;
    }

    if(this->callTable[5]>0)
    {
        int times = this->callTable[5];
        (this->*invMatFuncTable[5])(invMatrix,begin,times);
        begin = begin + times*36;
    }

    if(this->callTable[6]>0)
    {
        int times = this->callTable[6];
        (this->*invMatFuncTable[6])(invMatrix,begin,times);
        begin = begin + times*32;
    }

    if(this->callTable[7]>0)
    {
        int times = this->callTable[7];
        (this->*invMatFuncTable[7])(invMatrix,begin,times);
        begin = begin + times*28;
    }

    if(this->callTable[8]>0)
    {
        int times = this->callTable[8];
        (this->*invMatFuncTable[8])(invMatrix,begin,times);
        begin = begin + times*24;
    }

    if(this->callTable[9]>0)
    {
        int times = this->callTable[9];
        (this->*invMatFuncTable[9])(invMatrix,begin,times);
        begin = begin + times*20;
    }

    if(this->callTable[10]>0)
    {
        int times = this->callTable[10];
        (this->*invMatFuncTable[10])(invMatrix,begin,times);
        begin = begin + times*16;
    }

    if(this->callTable[11]>0)
    {
        int times = this->callTable[11];
        (this->*invMatFuncTable[11])(invMatrix,begin,times);
        begin = begin + times*12;
    }

    if(this->callTable[12]>0)
    {
        int times = this->callTable[12];
        (this->*invMatFuncTable[12])(invMatrix,begin,times);
        begin = begin + times*8;
    }

    if(this->callTable[13]>0)
    {
        int times = this->callTable[13];
        (this->*invMatFuncTable[13])(invMatrix,begin,times);
        begin = begin + times*4;
    }

    if(remain==3)
    {
        this->invMatrix3(invMatrix,begin,1);
    }
    else if(remain==2)
    {
        this->invMatrix2(invMatrix,begin,1);
    }
    else if(remain==1)
    {
        this->invMatrix1(invMatrix,begin,1);
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*56) columns
void LUsolver::invMatrix56
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double* __restrict__ b = this->invMatRowPtr[i];
            double* __restrict__ b = &invMatrix[i*this->N];
            __m256d sum0v = load256d(&b[k*56+0+begin]);
            __m256d sum4v = load256d(&b[k*56+4+begin]);
            __m256d sum8v = load256d(&b[k*56+8+begin]);
            __m256d sum12v = load256d(&b[k*56+12+begin]);
            __m256d sum16v = load256d(&b[k*56+16+begin]);
            __m256d sum20v = load256d(&b[k*56+20+begin]);
            __m256d sum24v = load256d(&b[k*56+24+begin]);
            __m256d sum28v = load256d(&b[k*56+28+begin]);
            __m256d sum32v = load256d(&b[k*56+32+begin]);
            __m256d sum36v = load256d(&b[k*56+36+begin]);
            __m256d sum40v = load256d(&b[k*56+40+begin]);
            __m256d sum44v = load256d(&b[k*56+44+begin]);
            __m256d sum48v = load256d(&b[k*56+48+begin]);
            __m256d sum52v = load256d(&b[k*56+52+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                //double* __restrict__ bj = this->invMatRowPtr[j];
                double* __restrict__ bj = &invMatrix[j*this->N];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*this->N+j]);

                __m256d bj0v = load256d(&bj[k*56+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*56+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*56+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*56+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*56+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*56+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*56+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*56+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*56+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*56+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*56+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*56+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);

                __m256d bj48v = load256d(&bj[k*56+48+begin]);
                sum48v = _mm256_fmadd_pd(vv,bj48v,sum48v);

                __m256d bj52v = load256d(&bj[k*56+52+begin]);
                sum52v = _mm256_fmadd_pd(vv,bj52v,sum52v);
            }
            store256d(&b[k*56+0+begin],sum0v);
            store256d(&b[k*56+4+begin],sum4v);
            store256d(&b[k*56+8+begin],sum8v);
            store256d(&b[k*56+12+begin],sum12v);
            store256d(&b[k*56+16+begin],sum16v);
            store256d(&b[k*56+20+begin],sum20v);
            store256d(&b[k*56+24+begin],sum24v);

            store256d(&b[k*56+28+begin],sum28v);
            store256d(&b[k*56+32+begin],sum32v);
            store256d(&b[k*56+36+begin],sum36v);
            store256d(&b[k*56+40+begin],sum40v);
            store256d(&b[k*56+44+begin],sum44v);
            store256d(&b[k*56+48+begin],sum48v);
            store256d(&b[k*56+52+begin],sum52v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double* __restrict__ b = this->invMatRowPtr[i];
            double* __restrict__ b = &invMatrix[i*this->N];
            __m256d sum0v = load256d(&b[k*56+0+begin]);
            __m256d sum4v = load256d(&b[k*56+4+begin]);
            __m256d sum8v = load256d(&b[k*56+8+begin]);
            __m256d sum12v = load256d(&b[k*56+12+begin]);
            __m256d sum16v = load256d(&b[k*56+16+begin]);
            __m256d sum20v = load256d(&b[k*56+20+begin]);
            __m256d sum24v = load256d(&b[k*56+24+begin]);
            __m256d sum28v = load256d(&b[k*56+28+begin]);
            __m256d sum32v = load256d(&b[k*56+32+begin]);
            __m256d sum36v = load256d(&b[k*56+36+begin]);
            __m256d sum40v = load256d(&b[k*56+40+begin]);
            __m256d sum44v = load256d(&b[k*56+44+begin]);
            __m256d sum48v = load256d(&b[k*56+48+begin]);
            __m256d sum52v = load256d(&b[k*56+52+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                //double* __restrict__ bj = this->invMatRowPtr[j];
                double* __restrict__ bj = &invMatrix[j*this->N];
                __m256d vv = _mm256_set1_pd(-v_[i*this->N+j]);

                __m256d bj0v = load256d(&bj[k*56+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*56+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*56+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*56+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*56+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*56+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*56+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*56+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*56+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*56+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*56+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*56+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);

                __m256d bj48v = load256d(&bj[k*56+48+begin]);
                sum48v = _mm256_fmadd_pd(vv,bj48v,sum48v);

                __m256d bj52v = load256d(&bj[k*56+52+begin]);
                sum52v = _mm256_fmadd_pd(vv,bj52v,sum52v);
            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*56+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*56+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*56+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*56+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*56+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*56+20+begin],sum20v);
            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*56+24+begin],sum24v);


            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*56+28+begin],sum28v);
            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*56+32+begin],sum32v);
            sum36v = _mm256_mul_pd(invDv,sum36v);
            store256d(&b[k*56+36+begin],sum36v);
            sum40v = _mm256_mul_pd(invDv,sum40v);
            store256d(&b[k*56+40+begin],sum40v);
            sum44v = _mm256_mul_pd(invDv,sum44v);
            store256d(&b[k*56+44+begin],sum44v);
            sum48v = _mm256_mul_pd(invDv,sum48v);
            store256d(&b[k*56+48+begin],sum48v);
            sum52v = _mm256_mul_pd(invDv,sum52v);
            store256d(&b[k*56+52+begin],sum52v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*52) columns
void LUsolver::invMatrix52
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*52+0+begin]);
            __m256d sum4v = load256d(&b[k*52+4+begin]);
            __m256d sum8v = load256d(&b[k*52+8+begin]);
            __m256d sum12v = load256d(&b[k*52+12+begin]);
            __m256d sum16v = load256d(&b[k*52+16+begin]);
            __m256d sum20v = load256d(&b[k*52+20+begin]);
            __m256d sum24v = load256d(&b[k*52+24+begin]);
            __m256d sum28v = load256d(&b[k*52+28+begin]);
            __m256d sum32v = load256d(&b[k*52+32+begin]);
            __m256d sum36v = load256d(&b[k*52+36+begin]);
            __m256d sum40v = load256d(&b[k*52+40+begin]);
            __m256d sum44v = load256d(&b[k*52+44+begin]);
            __m256d sum48v = load256d(&b[k*52+48+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*52+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*52+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*52+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*52+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*52+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*52+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*52+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*52+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*52+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*52+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*52+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*52+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);

                __m256d bj48v = load256d(&bj[k*52+48+begin]);
                sum48v = _mm256_fmadd_pd(vv,bj48v,sum48v);
            }
            store256d(&b[k*52+0+begin],sum0v);
            store256d(&b[k*52+4+begin],sum4v);
            store256d(&b[k*52+8+begin],sum8v);
            store256d(&b[k*52+12+begin],sum12v);
            store256d(&b[k*52+16+begin],sum16v);
            store256d(&b[k*52+20+begin],sum20v);
            store256d(&b[k*52+24+begin],sum24v);

            store256d(&b[k*52+28+begin],sum28v);
            store256d(&b[k*52+32+begin],sum32v);
            store256d(&b[k*52+36+begin],sum36v);
            store256d(&b[k*52+40+begin],sum40v);
            store256d(&b[k*52+44+begin],sum44v);
            store256d(&b[k*52+48+begin],sum48v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*52+0+begin]);
            __m256d sum4v = load256d(&b[k*52+4+begin]);
            __m256d sum8v = load256d(&b[k*52+8+begin]);
            __m256d sum12v = load256d(&b[k*52+12+begin]);
            __m256d sum16v = load256d(&b[k*52+16+begin]);
            __m256d sum20v = load256d(&b[k*52+20+begin]);
            __m256d sum24v = load256d(&b[k*52+24+begin]);
            __m256d sum28v = load256d(&b[k*52+28+begin]);
            __m256d sum32v = load256d(&b[k*52+32+begin]);
            __m256d sum36v = load256d(&b[k*52+36+begin]);
            __m256d sum40v = load256d(&b[k*52+40+begin]);
            __m256d sum44v = load256d(&b[k*52+44+begin]);
            __m256d sum48v = load256d(&b[k*52+48+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*52+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*52+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*52+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*52+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*52+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*52+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*52+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*52+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*52+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*52+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*52+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*52+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);

                __m256d bj48v = load256d(&bj[k*52+48+begin]);
                sum48v = _mm256_fmadd_pd(vv,bj48v,sum48v);
            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*52+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*52+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*52+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*52+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*52+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*52+20+begin],sum20v);
            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*52+24+begin],sum24v);


            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*52+28+begin],sum28v);
            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*52+32+begin],sum32v);
            sum36v = _mm256_mul_pd(invDv,sum36v);
            store256d(&b[k*52+36+begin],sum36v);
            sum40v = _mm256_mul_pd(invDv,sum40v);
            store256d(&b[k*52+40+begin],sum40v);
            sum44v = _mm256_mul_pd(invDv,sum44v);
            store256d(&b[k*52+44+begin],sum44v);
            sum48v = _mm256_mul_pd(invDv,sum48v);
            store256d(&b[k*52+48+begin],sum48v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*48) columns
void LUsolver::invMatrix48
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*48+0+begin]);
            __m256d sum4v = load256d(&b[k*48+4+begin]);
            __m256d sum8v = load256d(&b[k*48+8+begin]);
            __m256d sum12v = load256d(&b[k*48+12+begin]);
            __m256d sum16v = load256d(&b[k*48+16+begin]);
            __m256d sum20v = load256d(&b[k*48+20+begin]);
            __m256d sum24v = load256d(&b[k*48+24+begin]);
            __m256d sum28v = load256d(&b[k*48+28+begin]);
            __m256d sum32v = load256d(&b[k*48+32+begin]);
            __m256d sum36v = load256d(&b[k*48+36+begin]);
            __m256d sum40v = load256d(&b[k*48+40+begin]);
            __m256d sum44v = load256d(&b[k*48+44+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*48+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*48+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*48+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*48+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*48+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*48+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*48+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*48+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*48+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*48+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*48+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*48+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);
            }
            store256d(&b[k*48+0+begin],sum0v);
            store256d(&b[k*48+4+begin],sum4v);
            store256d(&b[k*48+8+begin],sum8v);
            store256d(&b[k*48+12+begin],sum12v);
            store256d(&b[k*48+16+begin],sum16v);
            store256d(&b[k*48+20+begin],sum20v);
            store256d(&b[k*48+24+begin],sum24v);

            store256d(&b[k*48+28+begin],sum28v);
            store256d(&b[k*48+32+begin],sum32v);
            store256d(&b[k*48+36+begin],sum36v);
            store256d(&b[k*48+40+begin],sum40v);
            store256d(&b[k*48+44+begin],sum44v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*48+0+begin]);
            __m256d sum4v = load256d(&b[k*48+4+begin]);
            __m256d sum8v = load256d(&b[k*48+8+begin]);
            __m256d sum12v = load256d(&b[k*48+12+begin]);
            __m256d sum16v = load256d(&b[k*48+16+begin]);
            __m256d sum20v = load256d(&b[k*48+20+begin]);
            __m256d sum24v = load256d(&b[k*48+24+begin]);
            __m256d sum28v = load256d(&b[k*48+28+begin]);
            __m256d sum32v = load256d(&b[k*48+32+begin]);
            __m256d sum36v = load256d(&b[k*48+36+begin]);
            __m256d sum40v = load256d(&b[k*48+40+begin]);
            __m256d sum44v = load256d(&b[k*48+44+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*48+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*48+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*48+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*48+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*48+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*48+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*48+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*48+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*48+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*48+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*48+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);

                __m256d bj44v = load256d(&bj[k*48+44+begin]);
                sum44v = _mm256_fmadd_pd(vv,bj44v,sum44v);
            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*48+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*48+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*48+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*48+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*48+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*48+20+begin],sum20v);


            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*48+28+begin],sum28v);
            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*48+32+begin],sum32v);
            sum36v = _mm256_mul_pd(invDv,sum36v);
            store256d(&b[k*48+36+begin],sum36v);
            sum40v = _mm256_mul_pd(invDv,sum40v);
            store256d(&b[k*48+40+begin],sum40v);
            sum44v = _mm256_mul_pd(invDv,sum44v);
            store256d(&b[k*48+44+begin],sum44v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*44) columns
void LUsolver::invMatrix44
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*44+0+begin]);
            __m256d sum4v = load256d(&b[k*44+4+begin]);
            __m256d sum8v = load256d(&b[k*44+8+begin]);
            __m256d sum12v = load256d(&b[k*44+12+begin]);
            __m256d sum16v = load256d(&b[k*44+16+begin]);
            __m256d sum20v = load256d(&b[k*44+20+begin]);
            __m256d sum24v = load256d(&b[k*44+24+begin]);
            __m256d sum28v = load256d(&b[k*44+28+begin]);
            __m256d sum32v = load256d(&b[k*44+32+begin]);
            __m256d sum36v = load256d(&b[k*44+36+begin]);
            __m256d sum40v = load256d(&b[k*44+40+begin]);


            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*44+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*44+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*44+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*44+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*44+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*44+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*44+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*44+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*44+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*44+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*44+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);


            }
            store256d(&b[k*44+0+begin],sum0v);
            store256d(&b[k*44+4+begin],sum4v);
            store256d(&b[k*44+8+begin],sum8v);
            store256d(&b[k*44+12+begin],sum12v);
            store256d(&b[k*44+16+begin],sum16v);
            store256d(&b[k*44+20+begin],sum20v);
            store256d(&b[k*44+24+begin],sum24v);

            store256d(&b[k*44+28+begin],sum28v);
            store256d(&b[k*44+32+begin],sum32v);
            store256d(&b[k*44+36+begin],sum36v);
            store256d(&b[k*44+40+begin],sum40v);

        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*44+0+begin]);
            __m256d sum4v = load256d(&b[k*44+4+begin]);
            __m256d sum8v = load256d(&b[k*44+8+begin]);
            __m256d sum12v = load256d(&b[k*44+12+begin]);
            __m256d sum16v = load256d(&b[k*44+16+begin]);
            __m256d sum20v = load256d(&b[k*44+20+begin]);
            __m256d sum24v = load256d(&b[k*44+24+begin]);
            __m256d sum28v = load256d(&b[k*44+28+begin]);
            __m256d sum32v = load256d(&b[k*44+32+begin]);
            __m256d sum36v = load256d(&b[k*44+36+begin]);
            __m256d sum40v = load256d(&b[k*44+40+begin]);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*44+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*44+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*44+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*44+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*44+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*44+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*44+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*44+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*44+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*44+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

                __m256d bj40v = load256d(&bj[k*44+40+begin]);
                sum40v = _mm256_fmadd_pd(vv,bj40v,sum40v);


            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*44+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*44+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*44+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*44+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*44+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*44+20+begin],sum20v);

            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*44+24+begin],sum24v);

            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*44+28+begin],sum28v);
            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*44+32+begin],sum32v);
            sum36v = _mm256_mul_pd(invDv,sum36v);
            store256d(&b[k*44+36+begin],sum36v);
            sum40v = _mm256_mul_pd(invDv,sum40v);
            store256d(&b[k*44+40+begin],sum40v);

        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*40) columns
void LUsolver::invMatrix40
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*40+0+begin]);
            __m256d sum4v = load256d(&b[k*40+4+begin]);
            __m256d sum8v = load256d(&b[k*40+8+begin]);
            __m256d sum12v = load256d(&b[k*40+12+begin]);
            __m256d sum16v = load256d(&b[k*40+16+begin]);
            __m256d sum20v = load256d(&b[k*40+20+begin]);
            __m256d sum24v = load256d(&b[k*40+24+begin]);
            __m256d sum28v = load256d(&b[k*40+28+begin]);
            __m256d sum32v = load256d(&b[k*40+32+begin]);
            __m256d sum36v = load256d(&b[k*40+36+begin]);



            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*40+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*40+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*40+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*40+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*40+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*40+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*40+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*40+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*40+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*40+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);




            }
            store256d(&b[k*40+0+begin],sum0v);
            store256d(&b[k*40+4+begin],sum4v);
            store256d(&b[k*40+8+begin],sum8v);
            store256d(&b[k*40+12+begin],sum12v);
            store256d(&b[k*40+16+begin],sum16v);
            store256d(&b[k*40+20+begin],sum20v);
            store256d(&b[k*40+24+begin],sum24v);

            store256d(&b[k*40+28+begin],sum28v);
            store256d(&b[k*40+32+begin],sum32v);
            store256d(&b[k*40+36+begin],sum36v);


        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*40+0+begin]);
            __m256d sum4v = load256d(&b[k*40+4+begin]);
            __m256d sum8v = load256d(&b[k*40+8+begin]);
            __m256d sum12v = load256d(&b[k*40+12+begin]);
            __m256d sum16v = load256d(&b[k*40+16+begin]);
            __m256d sum20v = load256d(&b[k*40+20+begin]);
            __m256d sum24v = load256d(&b[k*40+24+begin]);
            __m256d sum28v = load256d(&b[k*40+28+begin]);
            __m256d sum32v = load256d(&b[k*40+32+begin]);
            __m256d sum36v = load256d(&b[k*40+36+begin]);


            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*40+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*40+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*40+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*40+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*40+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*40+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*40+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*40+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*40+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

                __m256d bj36v = load256d(&bj[k*40+36+begin]);
                sum36v = _mm256_fmadd_pd(vv,bj36v,sum36v);

            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*40+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*40+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*40+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*40+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*40+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*40+20+begin],sum20v);
            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*40+24+begin],sum24v);


            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*40+28+begin],sum28v);
            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*40+32+begin],sum32v);
            sum36v = _mm256_mul_pd(invDv,sum36v);
            store256d(&b[k*40+36+begin],sum36v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*36) columns
void LUsolver::invMatrix36
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*36+0+begin]);
            __m256d sum4v = load256d(&b[k*36+4+begin]);
            __m256d sum8v = load256d(&b[k*36+8+begin]);
            __m256d sum12v = load256d(&b[k*36+12+begin]);
            __m256d sum16v = load256d(&b[k*36+16+begin]);
            __m256d sum20v = load256d(&b[k*36+20+begin]);
            __m256d sum24v = load256d(&b[k*36+24+begin]);
            __m256d sum28v = load256d(&b[k*36+28+begin]);
            __m256d sum32v = load256d(&b[k*36+32+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*36+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*36+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*36+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*36+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*36+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*36+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*36+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*36+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*36+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

            }
            store256d(&b[k*36+0+begin],sum0v);
            store256d(&b[k*36+4+begin],sum4v);
            store256d(&b[k*36+8+begin],sum8v);
            store256d(&b[k*36+12+begin],sum12v);
            store256d(&b[k*36+16+begin],sum16v);
            store256d(&b[k*36+20+begin],sum20v);
            store256d(&b[k*36+24+begin],sum24v);

            store256d(&b[k*36+28+begin],sum28v);
            store256d(&b[k*36+32+begin],sum32v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*36+0+begin]);
            __m256d sum4v = load256d(&b[k*36+4+begin]);
            __m256d sum8v = load256d(&b[k*36+8+begin]);
            __m256d sum12v = load256d(&b[k*36+12+begin]);
            __m256d sum16v = load256d(&b[k*36+16+begin]);
            __m256d sum20v = load256d(&b[k*36+20+begin]);
            __m256d sum24v = load256d(&b[k*36+24+begin]);
            __m256d sum28v = load256d(&b[k*36+28+begin]);
            __m256d sum32v = load256d(&b[k*36+32+begin]);


            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*36+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*36+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*36+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*36+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*36+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*36+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*36+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*36+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

                __m256d bj32v = load256d(&bj[k*36+32+begin]);
                sum32v = _mm256_fmadd_pd(vv,bj32v,sum32v);

            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*36+0+begin],sum0v);

            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*36+4+begin],sum4v);

            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*36+8+begin],sum8v);

            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*36+12+begin],sum12v);

            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*36+16+begin],sum16v);

            sum20v = _mm256_mul_pd(invDv,sum20v);     
            store256d(&b[k*36+20+begin],sum20v);

            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*36+24+begin],sum24v);

            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*36+28+begin],sum28v);

            sum32v = _mm256_mul_pd(invDv,sum32v);
            store256d(&b[k*36+32+begin],sum32v);

        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*36) columns
void LUsolver::invMatrix32
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*32+0+begin]);
            __m256d sum4v = load256d(&b[k*32+4+begin]);
            __m256d sum8v = load256d(&b[k*32+8+begin]);
            __m256d sum12v = load256d(&b[k*32+12+begin]);
            __m256d sum16v = load256d(&b[k*32+16+begin]);
            __m256d sum20v = load256d(&b[k*32+20+begin]);
            __m256d sum24v = load256d(&b[k*32+24+begin]);
            __m256d sum28v = load256d(&b[k*32+28+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*32+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*32+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*32+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*32+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*32+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*32+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*32+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*32+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);
            }
            store256d(&b[k*32+0+begin],sum0v);
            store256d(&b[k*32+4+begin],sum4v);
            store256d(&b[k*32+8+begin],sum8v);
            store256d(&b[k*32+12+begin],sum12v);
            store256d(&b[k*32+16+begin],sum16v);
            store256d(&b[k*32+20+begin],sum20v);
            store256d(&b[k*32+24+begin],sum24v);
            store256d(&b[k*32+28+begin],sum28v);

        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*32+0+begin]);
            __m256d sum4v = load256d(&b[k*32+4+begin]);
            __m256d sum8v = load256d(&b[k*32+8+begin]);
            __m256d sum12v = load256d(&b[k*32+12+begin]);
            __m256d sum16v = load256d(&b[k*32+16+begin]);
            __m256d sum20v = load256d(&b[k*32+20+begin]);
            __m256d sum24v = load256d(&b[k*32+24+begin]);
            __m256d sum28v = load256d(&b[k*32+28+begin]);



            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*32+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*32+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*32+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*32+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*32+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*32+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*32+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

                __m256d bj28v = load256d(&bj[k*32+28+begin]);
                sum28v = _mm256_fmadd_pd(vv,bj28v,sum28v);

            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*32+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*32+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*32+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*32+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*32+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*32+20+begin],sum20v);
            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*32+24+begin],sum24v);
            sum28v = _mm256_mul_pd(invDv,sum28v);
            store256d(&b[k*32+28+begin],sum28v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*28) columns
void LUsolver::invMatrix28
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*28+0+begin]);
            __m256d sum4v = load256d(&b[k*28+4+begin]);
            __m256d sum8v = load256d(&b[k*28+8+begin]);
            __m256d sum12v = load256d(&b[k*28+12+begin]);
            __m256d sum16v = load256d(&b[k*28+16+begin]);
            __m256d sum20v = load256d(&b[k*28+20+begin]);
            __m256d sum24v = load256d(&b[k*28+24+begin]);


            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*28+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*28+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*28+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*28+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*28+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*28+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*28+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);
            }
            store256d(&b[k*28+0+begin],sum0v);
            store256d(&b[k*28+4+begin],sum4v);
            store256d(&b[k*28+8+begin],sum8v);
            store256d(&b[k*28+12+begin],sum12v);
            store256d(&b[k*28+16+begin],sum16v);
            store256d(&b[k*28+20+begin],sum20v);
            store256d(&b[k*28+24+begin],sum24v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*28+0+begin]);
            __m256d sum4v = load256d(&b[k*28+4+begin]);
            __m256d sum8v = load256d(&b[k*28+8+begin]);
            __m256d sum12v = load256d(&b[k*28+12+begin]);
            __m256d sum16v = load256d(&b[k*28+16+begin]);
            __m256d sum20v = load256d(&b[k*28+20+begin]);
            __m256d sum24v = load256d(&b[k*28+24+begin]);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*28+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*28+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*28+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*28+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*28+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*28+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

                __m256d bj24v = load256d(&bj[k*28+24+begin]);
                sum24v = _mm256_fmadd_pd(vv,bj24v,sum24v);

            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*28+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*28+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*28+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*28+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*28+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*28+20+begin],sum20v);
            sum24v = _mm256_mul_pd(invDv,sum24v);
            store256d(&b[k*28+24+begin],sum24v);
        }
    }
}


// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*24) columns
void LUsolver::invMatrix24
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*24+0+begin]);
            __m256d sum4v = load256d(&b[k*24+4+begin]);
            __m256d sum8v = load256d(&b[k*24+8+begin]);
            __m256d sum12v = load256d(&b[k*24+12+begin]);
            __m256d sum16v = load256d(&b[k*24+16+begin]);
            __m256d sum20v = load256d(&b[k*24+20+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*24+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*24+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*24+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*24+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*24+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*24+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);
            }
            store256d(&b[k*24+0+begin],sum0v);
            store256d(&b[k*24+4+begin],sum4v);
            store256d(&b[k*24+8+begin],sum8v);
            store256d(&b[k*24+12+begin],sum12v);
            store256d(&b[k*24+16+begin],sum16v);
            store256d(&b[k*24+20+begin],sum20v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*24+0+begin]);
            __m256d sum4v = load256d(&b[k*24+4+begin]);
            __m256d sum8v = load256d(&b[k*24+8+begin]);
            __m256d sum12v = load256d(&b[k*24+12+begin]);
            __m256d sum16v = load256d(&b[k*24+16+begin]);
            __m256d sum20v = load256d(&b[k*24+20+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*24+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*24+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*24+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*24+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*24+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

                __m256d bj20v = load256d(&bj[k*24+20+begin]);
                sum20v = _mm256_fmadd_pd(vv,bj20v,sum20v);

            }
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*24+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*24+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*24+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*24+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*24+16+begin],sum16v);
            sum20v = _mm256_mul_pd(invDv,sum20v);
            store256d(&b[k*24+20+begin],sum20v);
        }
    }
}


// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*20) columns 
void LUsolver::invMatrix20
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*20+0+begin]);
            __m256d sum4v = load256d(&b[k*20+4+begin]);
            __m256d sum8v = load256d(&b[k*20+8+begin]);
            __m256d sum12v = load256d(&b[k*20+12+begin]);
            __m256d sum16v = load256d(&b[k*20+16+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*20+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*20+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*20+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*20+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*20+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);

            }
            store256d(&b[k*20+0+begin],sum0v);
            store256d(&b[k*20+4+begin],sum4v);
            store256d(&b[k*20+8+begin],sum8v);
            store256d(&b[k*20+12+begin],sum12v);
            store256d(&b[k*20+16+begin],sum16v);
        }


    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*20+0+begin]);
            __m256d sum4v = load256d(&b[k*20+4+begin]);
            __m256d sum8v = load256d(&b[k*20+8+begin]);
            __m256d sum12v = load256d(&b[k*20+12+begin]);
            __m256d sum16v = load256d(&b[k*20+16+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*20+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*20+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*20+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*20+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

                __m256d bj16v = load256d(&bj[k*20+16+begin]);
                sum16v = _mm256_fmadd_pd(vv,bj16v,sum16v);
            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*20+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*20+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*20+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*20+12+begin],sum12v);
            sum16v = _mm256_mul_pd(invDv,sum16v);
            store256d(&b[k*20+16+begin],sum16v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*16) columns 
void LUsolver::invMatrix16
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*16+0+begin]);
            __m256d sum4v = load256d(&b[k*16+4+begin]);
            __m256d sum8v = load256d(&b[k*16+8+begin]);
            __m256d sum12v = load256d(&b[k*16+12+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*16+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*16+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*16+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*16+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);

            }
            store256d(&b[k*16+0+begin],sum0v);
            store256d(&b[k*16+4+begin],sum4v);
            store256d(&b[k*16+8+begin],sum8v);
            store256d(&b[k*16+12+begin],sum12v);
        }


    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*16+0+begin]);
            __m256d sum4v = load256d(&b[k*16+4+begin]);
            __m256d sum8v = load256d(&b[k*16+8+begin]);
            __m256d sum12v = load256d(&b[k*16+12+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*16+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*16+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*16+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);

                __m256d bj12v = load256d(&bj[k*16+12+begin]);
                sum12v = _mm256_fmadd_pd(vv,bj12v,sum12v);
            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*16+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*16+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*16+8+begin],sum8v);
            sum12v = _mm256_mul_pd(invDv,sum12v);
            store256d(&b[k*16+12+begin],sum12v);
        }
    }
}


// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*12) columns 
void LUsolver::invMatrix12
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*12+0+begin]);
            __m256d sum4v = load256d(&b[k*12+4+begin]);
            __m256d sum8v = load256d(&b[k*12+8+begin]);

            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*12+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*12+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*12+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);
            }
            store256d(&b[k*12+0+begin],sum0v);
            store256d(&b[k*12+4+begin],sum4v);
            store256d(&b[k*12+8+begin],sum8v);
        }


    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*12+0+begin]);
            __m256d sum4v = load256d(&b[k*12+4+begin]);
            __m256d sum8v = load256d(&b[k*12+8+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*12+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*12+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

                __m256d bj8v = load256d(&bj[k*12+8+begin]);
                sum8v = _mm256_fmadd_pd(vv,bj8v,sum8v);
            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*12+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*12+4+begin],sum4v);
            sum8v = _mm256_mul_pd(invDv,sum8v);
            store256d(&b[k*12+8+begin],sum8v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*8) columns 
void LUsolver::invMatrix8
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*8+0+begin]);
            __m256d sum4v = load256d(&b[k*8+4+begin]);


            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*8+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*8+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);

            }
            store256d(&b[k*8+0+begin],sum0v);
            store256d(&b[k*8+4+begin],sum4v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*8+0+begin]);
            __m256d sum4v = load256d(&b[k*8+4+begin]);
            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*8+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);

                __m256d bj4v = load256d(&bj[k*8+4+begin]);
                sum4v = _mm256_fmadd_pd(vv,bj4v,sum4v);
            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*8+0+begin],sum0v);
            sum4v = _mm256_mul_pd(invDv,sum4v);
            store256d(&b[k*8+4+begin],sum4v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*4) columns 
void LUsolver::invMatrix4
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m256d sum0v = load256d(&b[k*4+0+begin]);



            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*4+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);


            }
            store256d(&b[k*4+0+begin],sum0v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m256d sum0v = load256d(&b[k*4+0+begin]);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                __m256d bj0v = load256d(&bj[k*4+0+begin]);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);


            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            store256d(&b[k*4+0+begin],sum0v);
        }
    }
}



// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*3) columns 
void LUsolver::invMatrix3
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            //__m256d sum0v = load256d(&b[k*4+0+begin]);
            __m256d sum0v = _mm256_setr_pd(b[k*3+0+begin],b[k*3+1+begin],b[k*3+2+begin],0);



            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);


                //__m256d bj0v = load256d(&bj[k*4+0+begin]);
                __m256d bj0v = _mm256_setr_pd(bj[k*3+0+begin],bj[k*3+1+begin],bj[k*3+2+begin],0);

                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);


            }
            //store256d(&b[k*4+0+begin],sum0v);
            b[k*3+0+begin] = this->get_elem0(sum0v);
            b[k*3+1+begin] = this->get_elem1(sum0v);
            b[k*3+2+begin] = this->get_elem2(sum0v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            //__m256d sum0v = load256d(&b[k*4+0+begin]);
            __m256d sum0v = _mm256_setr_pd(b[k*3+0+begin],b[k*3+1+begin],b[k*3+2+begin],0);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m256d vv = _mm256_set1_pd(-v_[i*alignN+j]);

                //__m256d bj0v = load256d(&bj[k*4+0+begin]);
                __m256d bj0v = _mm256_setr_pd(bj[k*3+0+begin],bj[k*3+1+begin],bj[k*3+2+begin],0);
                sum0v = _mm256_fmadd_pd(vv,bj0v,sum0v);


            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m256d invDv = _mm256_set1_pd(this->invD[i]);
            sum0v = _mm256_mul_pd(invDv,sum0v);
            //store256d(&b[k*4+0+begin],sum0v);
            b[k*3+0+begin] = this->get_elem0(sum0v);
            b[k*3+1+begin] = this->get_elem1(sum0v);
            b[k*3+2+begin] = this->get_elem2(sum0v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*2) columns 
void LUsolver::invMatrix2
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            __m128d sum0v = load128d(&b[k*2+0+begin]);


            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                __m128d vv = _mm_set1_pd(-v_[i*alignN+j]);

                __m128d bj0v = load128d(&bj[k*2+0+begin]);
                sum0v = _mm_fmadd_pd(vv,bj0v,sum0v);


            }
            store128d(&b[k*2+0+begin],sum0v);
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            __m128d sum0v = load128d(&b[k*2+0+begin]);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                __m128d vv = _mm_set1_pd(-v_[i*alignN+j]);

                __m128d bj0v = load128d(&bj[k*2+0+begin]);
                sum0v = _mm_fmadd_pd(vv,bj0v,sum0v);


            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            __m128d invDv = _mm_set1_pd(this->invD[i]);
            sum0v = _mm_mul_pd(invDv,sum0v);
            store128d(&b[k*2+0+begin],sum0v);
        }
    }
}

// get inverse of v by solving L*U*invA=I
// only get the results between [begin,begin+times*1) columns 
void LUsolver::invMatrix1
(
    double* __restrict__ invMatrix,
    const int begin,
    const int times
)
{

    for(unsigned int i = 0; i < this->N; i++)
    {
        for(int k = 0; k < times; k=k+1)
        {
            double* __restrict__ b = this->invMatRowPtr[i];
            double sum0v = (b[k*1+0+begin]);


            for(unsigned int j = 0; j < i; j++)
            {
                double* __restrict__ bj = this->invMatRowPtr[j];

                //sum = sum - v_[i*alignN+j]*b[j];
                double vv = (-v_[i*alignN+j]);

                double bj0v = (bj[k*1+0+begin]);

                sum0v = sum0v + vv*bj0v;
            }
            b[k*1+0+begin]=sum0v;
        }
    }

    for(int i = N-1; i >= 0; i--)
    {
        for(int k = 0; k < times; k=k+1)
        {
            //double sum = b[i];
            //get pointer for i-th row
            double* __restrict__ b = this->invMatRowPtr[i];

            double sum0v = (b[k*1+0+begin]);

            for(unsigned int j = i+1; j<this->N; j++)
            {
                //const double xj = b[j];
                //sum = sum - v_[i*alignN+j]*xj;

                //get j-th row
                double* __restrict__ bj = this->invMatRowPtr[j];

                double vv = (-v_[i*alignN+j]);

                double bj0v = (bj[k*1+0+begin]);
                //sum0v = _mm_fmadd_pd(vv,bj0v,sum0v);
                sum0v = sum0v + vv*bj0v;

            }
            //b[i] = sum/v_[i*alignN+i];
            //b[i] = sum*this->invD[i];
            double invDv = (this->invD[i]);
            sum0v = (invDv*sum0v);
            b[k*1+0+begin]=sum0v;
        }
    }
}


}