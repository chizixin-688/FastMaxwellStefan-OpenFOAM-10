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
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <array>

//---------------------------------
// 2. SIMD headers
//---------------------------------
#include <immintrin.h>

//---------------------------------
// 3. OpenFOAM-10 headers
//---------------------------------
#include "speciesTable.H"

//---------------------------------
// 4. FastMaxwellStefan headers
//---------------------------------
#include "Macro.H"
#include "vec_math_avx2.H"

//=============================================================================//
#ifndef CKDDreader_H
#define CKDDreader_H

namespace Foam
{

/*---------------------------------------------------------------------------*\
                Class CKDDreader Declaration
\*---------------------------------------------------------------------------*/

class CKDDreader
{
    // Private data

private:

    std::vector<std::array<double,4>> CoeffTable;

    std::vector<double> result;

    std::vector<std::pair<std::string, std::string>> speciesPair;

public:


    // Constructors

        CKDDreader(const speciesTable& species,bool& useLogPolynomial);

        CKDDreader()=default;

        std::vector<std::string> speciesTable_;
        
        void readFile(int nSpecies,bool& useLogPolynomial);

        void getDD(double* DD,double logT, double p, unsigned int nSpecies);

        void getLogT(const double* __restrict__ T, double* __restrict__ logT, int cells);

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
};



}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif