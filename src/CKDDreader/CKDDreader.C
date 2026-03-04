/*---------------------------------------------------------------------------*\
  Copyright (C) 2026 Zixin Chi
  All rights reserved.

  This file is an original work developed independently by Zixin Chi.

  This program is distributed under the terms of the
  GNU General Public License v3.0.

  Description
      This file is used for calculating binary diffusion coefficient
      using log polynomial function of CHEMKIN format

  Author
      Zixin Chi <chizixin@buaa.edu.cn>

\*---------------------------------------------------------------------------*/

//=============================================================================//

//---------------------------------
// 1. Standard C++ library headers
//---------------------------------
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
//---------------------------------
// 2. FastChemistry headers
//---------------------------------
#include "CKDDreader.H"

//---------------------------------
// 3. SIMD / AVX2 headers
//---------------------------------
#include <immintrin.h>

//=============================================================================//
namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
CKDDreader::CKDDreader
(
    const speciesTable& species,
    bool& useLogPolynomial
)
{
    int nSpecies = species.size();
    this->speciesTable_.resize(nSpecies);
    for(label i=0; i<nSpecies; i++){this->speciesTable_[i] = species[i];}

    std::vector<std::string> transportFile;

    std::vector<std::string> lines;

    std::ifstream is("constant/CKtransport");

    if (!is.is_open())
    {
        useLogPolynomial = false;
        return;
    }
    std::string line;
    while (std::getline(is, line))
    {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            continue;

        auto last = line.find_last_not_of(" \t\r\n");
        lines.push_back(line.substr(first, last - first + 1));
    }
    transportFile = lines;

    // Find the line of COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS
    std::size_t idx = 0;
    std::string DDCoeffs = "COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS";
    auto it = std::find(transportFile.begin(), transportFile.end(), DDCoeffs);
    if(it == transportFile.end())
    {
        throw std::runtime_error("Cannot find the COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS");
        std::exit(0);
    }
    else if (it != transportFile.end())
    {
        idx = it - transportFile.begin();
        idx = idx+2;
    }

    // Convert std::string to double
    {
        std::size_t end = idx + nSpecies*(nSpecies-1)/2 + nSpecies ;
        for(std::size_t i = idx; i<end;i++)
        {
            std::istringstream iss(transportFile[i]);
            std::vector<std::string> words;
            std::string Word;
            while (iss >> Word) 
            {
                words.push_back(Word);
            }
            if(words.size()!=6){std::exit(0);}
            std::pair<std::string,std::string> pair(words[0],words[1]);
            this->speciesPair.push_back(pair);

            double a0 = std::stod(words[2]);
            double a1 = std::stod(words[3]);
            double a2 = std::stod(words[4]);
            double a3 = std::stod(words[5]);

            std::array<double,4> Coeffs={a0,a1,a2,a3};
            this->CoeffTable.push_back(Coeffs);
        }
    }

    //Reorder
    {
        unsigned k = 0;
        for(int i=0; i<nSpecies; i++)
        {
            for(int j=i; j<nSpecies; j++)
            {
                std::pair<std::string,std::string> targetPair0(this->speciesTable_[i],this->speciesTable_[j]);
                std::pair<std::string,std::string> targetPair1(this->speciesTable_[j],this->speciesTable_[i]);

                auto iter0 = std::find(this->speciesPair.begin(), this->speciesPair.end(), targetPair0);
                auto iter1 = std::find(this->speciesPair.begin(), this->speciesPair.end(), targetPair1);

                if(iter0==this->speciesPair.end() && iter1==this->speciesPair.end())
                {
                    std::string pair0 = this->speciesTable_[i]+"-"+this->speciesTable_[j];
                    std::string pair1 = this->speciesTable_[j]+"-"+this->speciesTable_[i];

                    std::cout<<"The string does not exist for either "<<pair0<<" or "<<pair1<<std::endl;
                    std::exit(0);
                }
                size_t index = -1;

                if(iter0!=this->speciesPair.end())
                {
                    index = std::distance(this->speciesPair.begin(), iter0);
                }
                else if(iter1!=this->speciesPair.end())
                {
                    index = std::distance(this->speciesPair.begin(), iter1);
                }
                std::swap(this->speciesPair[k],this->speciesPair[index]);
                std::swap(this->CoeffTable[k],this->CoeffTable[index]);
                k++;
            }
        }
    }

    result.resize(this->speciesPair.size());
    useLogPolynomial = true;
}


void CKDDreader::readFile(int nSpecies, bool& useLogPolynomial)
{
    std::vector<std::string> transportFile;

    std::vector<std::string> lines;

    std::ifstream is("constant/CKtransport");

    if (!is.is_open())
    {
        useLogPolynomial = false;
        return;
    }

    std::string line;
    while (std::getline(is, line))
    {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            continue;

        auto last = line.find_last_not_of(" \t\r\n");
        lines.push_back(line.substr(first, last - first + 1));
    }
    transportFile = lines;

    // Find the line of COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS
    std::size_t idx = 0;
    std::string DDCoeffs = "COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS";
    auto it = std::find(transportFile.begin(), transportFile.end(), DDCoeffs);
    if(it == transportFile.end())
    {
        throw std::runtime_error("Cannot find the COEFFICIENTS FOR SPECIES DIFFUSION COEFFICIENTS");
        std::exit(0);
    }
    else if (it != transportFile.end())
    {
        idx = it - transportFile.begin();
        idx = idx+2;
    }

    // Convert std::string to double
    {
        std::size_t end = idx + nSpecies*(nSpecies-1)/2 + nSpecies ;
        for(std::size_t i = idx; i<end;i++)
        {
            std::istringstream iss(transportFile[i]);
            std::vector<std::string> words;
            std::string Word;
            while (iss >> Word) 
            {
                words.push_back(Word);
            }
            if(words.size()!=6){std::exit(0);}
            std::pair<std::string,std::string> pair(words[0],words[1]);
            this->speciesPair.push_back(pair);

            double a0 = std::stod(words[2]);
            double a1 = std::stod(words[3]);
            double a2 = std::stod(words[4]);
            double a3 = std::stod(words[5]);

            std::array<double,4> Coeffs={a0,a1,a2,a3};
            this->CoeffTable.push_back(Coeffs);
        }
    }

    //Reorder
    {
        unsigned k = 0;
        for(int i=0; i<nSpecies; i++)
        {
            for(int j=i; j<nSpecies; j++)
            {
                std::pair<std::string,std::string> targetPair0(this->speciesTable_[i],this->speciesTable_[j]);
                std::pair<std::string,std::string> targetPair1(this->speciesTable_[j],this->speciesTable_[i]);

                auto iter0 = std::find(this->speciesPair.begin(), this->speciesPair.end(), targetPair0);
                auto iter1 = std::find(this->speciesPair.begin(), this->speciesPair.end(), targetPair1);

                if(iter0==this->speciesPair.end() && iter1==this->speciesPair.end())
                {
                    std::string pair0 = this->speciesTable_[i]+"-"+this->speciesTable_[j];
                    std::string pair1 = this->speciesTable_[j]+"-"+this->speciesTable_[i];

                    std::cout<<"The string does not exist for either "<<pair0<<" or "<<pair1<<std::endl;
                    std::exit(0);
                }
                size_t index = -1;

                if(iter0!=this->speciesPair.end())
                {
                    index = std::distance(this->speciesPair.begin(), iter0);
                }
                else if(iter1!=this->speciesPair.end())
                {
                    index = std::distance(this->speciesPair.begin(), iter1);
                }
                std::swap(this->speciesPair[k],this->speciesPair[index]);
                std::swap(this->CoeffTable[k],this->CoeffTable[index]);
                k++;
            }
        }
    }

    result.resize(this->speciesPair.size());
    useLogPolynomial = true;
}

// Compute Binary diffusion coefficient using log polynomial, the unit pressure in CHEMKIN is 101325Pa
void CKDDreader::getDD(double* DD, double logT, double p, unsigned int nSpecies)
{
    double scale = 10.1325/p;
    __m256d scalev = _mm256_set1_pd(scale);
    
    __m256d logTv = _mm256_set1_pd(logT);
    size_t remain = this->speciesPair.size()%4;
    for(size_t i=0; i<this->speciesPair.size()-remain;i=i+4)
    {
        double a2 = this->CoeffTable[i+0][2];
        double b2 = this->CoeffTable[i+1][2];
        double c2 = this->CoeffTable[i+2][2];
        double d2 = this->CoeffTable[i+3][2];
        __m256d A2 = _mm256_setr_pd(a2,b2,c2,d2);

        double a3 = this->CoeffTable[i+0][3];
        double b3 = this->CoeffTable[i+1][3];
        double c3 = this->CoeffTable[i+2][3];
        double d3 = this->CoeffTable[i+3][3];
        __m256d A3 = _mm256_setr_pd(a3,b3,c3,d3);
        __m256d logDDv = _mm256_fmadd_pd(A3,logTv,A2);

        double a1 = this->CoeffTable[i+0][1];
        double b1 = this->CoeffTable[i+1][1];
        double c1 = this->CoeffTable[i+2][1];
        double d1 = this->CoeffTable[i+3][1];
        __m256d A1 = _mm256_setr_pd(a1,b1,c1,d1);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A1);

        double a0 = this->CoeffTable[i+0][0];
        double b0 = this->CoeffTable[i+1][0];
        double c0 = this->CoeffTable[i+2][0];
        double d0 = this->CoeffTable[i+3][0];
        __m256d A0 = _mm256_setr_pd(a0,b0,c0,d0);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A0);
        __m256d DDv = vec256_expd(logDDv);
        DDv = _mm256_mul_pd(DDv,scalev);
        store256d(&this->result[i],DDv);
    }
    if(remain==1)
    {
        size_t i = this->speciesPair.size()-1;
        double a0 = this->CoeffTable[i+0][0];
        double a1 = this->CoeffTable[i+0][1];
        double a2 = this->CoeffTable[i+0][2];
        double a3 = this->CoeffTable[i+0][3];
        double logDD = (a0+logT*(a1+logT*(a2+a3*logT)));
        this->result[i] = std::exp(logDD)*scale;
    }
    else if(remain==2)
    {
        size_t i = this->speciesPair.size()-2;
        double a2 = this->CoeffTable[i+0][2];
        double b2 = this->CoeffTable[i+1][2];
        double c2 = this->CoeffTable[i+0][2];
        double d2 = this->CoeffTable[i+1][2];
        __m256d A2 = _mm256_setr_pd(a2,b2,c2,d2);

        double a3 = this->CoeffTable[i+0][3];
        double b3 = this->CoeffTable[i+1][3];
        double c3 = this->CoeffTable[i+0][3];
        double d3 = this->CoeffTable[i+1][3];
        __m256d A3 = _mm256_setr_pd(a3,b3,c3,d3);
        __m256d logDDv = _mm256_fmadd_pd(A3,logTv,A2);

        double a1 = this->CoeffTable[i+0][1];
        double b1 = this->CoeffTable[i+1][1];
        double c1 = this->CoeffTable[i+0][1];
        double d1 = this->CoeffTable[i+1][1];
        __m256d A1 = _mm256_setr_pd(a1,b1,c1,d1);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A1);

        double a0 = this->CoeffTable[i+0][0];
        double b0 = this->CoeffTable[i+1][0];
        double c0 = this->CoeffTable[i+0][0];
        double d0 = this->CoeffTable[i+1][0];
        __m256d A0 = _mm256_setr_pd(a0,b0,c0,d0);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A0);
        
        __m256d DDv = vec256_expd(logDDv);
        DDv = _mm256_mul_pd(DDv,scalev);
        store128d(&this->result[i],_mm256_castpd256_pd128(DDv));
    }
    else if(remain==3)
    {
        size_t i = this->speciesPair.size()-3;
        double a2 = this->CoeffTable[i+0][2];
        double b2 = this->CoeffTable[i+1][2];
        double c2 = this->CoeffTable[i+2][2];
        double d2 = this->CoeffTable[i+2][2];
        __m256d A2 = _mm256_setr_pd(a2,b2,c2,d2);

        double a3 = this->CoeffTable[i+0][3];
        double b3 = this->CoeffTable[i+1][3];
        double c3 = this->CoeffTable[i+2][3];
        double d3 = this->CoeffTable[i+2][3];
        __m256d A3 = _mm256_setr_pd(a3,b3,c3,d3);
        __m256d logDDv = _mm256_fmadd_pd(A3,logTv,A2);

        double a1 = this->CoeffTable[i+0][1];
        double b1 = this->CoeffTable[i+1][1];
        double c1 = this->CoeffTable[i+2][1];
        double d1 = this->CoeffTable[i+2][1];
        __m256d A1 = _mm256_setr_pd(a1,b1,c1,d1);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A1);

        double a0 = this->CoeffTable[i+0][0];
        double b0 = this->CoeffTable[i+1][0];
        double c0 = this->CoeffTable[i+2][0];
        double d0 = this->CoeffTable[i+2][0];
        __m256d A0 = _mm256_setr_pd(a0,b0,c0,d0);
        logDDv = _mm256_fmadd_pd(logDDv,logTv,A0);
        
        __m256d DDv = vec256_expd(logDDv);
        DDv = _mm256_mul_pd(DDv,scalev);
        this->result[i+0] = this->get_elem0(DDv);
        this->result[i+1] = this->get_elem1(DDv);
        this->result[i+2] = this->get_elem2(DDv);
    }
    int k = 0;
    for(size_t i=0; i<nSpecies; i++)
    {
        for(size_t j=0; j<nSpecies; j++)
        {
            if(j>=i)
            {
                DD[i*nSpecies+j] = this->result[k];
                k++;
            }
            else
            {
                DD[i*nSpecies+j] = DD[j*nSpecies+i];
            }
        }
    }
}

void CKDDreader::getLogT
(
    const double* __restrict__ T, 
    double* __restrict__ logT, 
    int times
)
{
    int remain = times%4;
    for(int i=0; i<times-remain; i=i+4)
    {
        __m256d Tv = load256d(&T[i]);
        __m256d logTv = vec256_logd(Tv);
        _mm256_storeu_pd(&logT[i],logTv);
    }
    if(remain==1)
    {
        int i = times-1;
        logT[i] = std::log(T[i]);
    }
    else if(remain==2)
    {
        int i = times-2;
        __m128d Tv = load128d(&T[i]);
        __m256d logTv = vec256_logd(_mm256_insertf128_pd (_mm256_castpd128_pd256 (Tv), Tv, 1));
        store128d(&logT[i],_mm256_castpd256_pd128(logTv));
    }
    else if(remain==3)
    {
        int i = times-3;
        __m256d Tv = _mm256_setr_pd(T[i],T[i+1],T[i+2],T[i+2]);
        __m256d logTv = vec256_logd(Tv);
        store128d(&logT[i],_mm256_castpd256_pd128(logTv));
        logT[i+0] = this->get_elem0(logTv);
        logT[i+1] = this->get_elem1(logTv);
        logT[i+2] = this->get_elem2(logTv);
    }
}
} // End namespace Foam
