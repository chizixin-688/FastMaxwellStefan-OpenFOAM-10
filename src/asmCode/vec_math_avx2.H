/*---------------------------------------------------------------------------*\
  Copyright (C) 2026 Zixin Chi
  All rights reserved.

  This file is an original work developed independently by Zixin Chi.

  This program is distributed under the terms of the
  GNU General Public License v3.0.

  Description
      This file is used to link vectorized math function from asmCode folder
      or GNU library.

  Author
      Zixin Chi <chizixin@buaa.edu.cn>

\*---------------------------------------------------------------------------*/

#ifndef VEC_MATH_AVX2_H
#define VEC_MATH_AVX2_H

//---------------------------------
// 2. SIMD headers
//---------------------------------
#include <immintrin.h>

//---------------------------------
// 4. FastMaxwellStefan headers
//---------------------------------
#include "Macro.H"

//=============================================================================//
namespace Foam
{
    #if USE_LOCALFILE_

        extern "C" 
        {
            __m256d _ZGVdN4v_exp_avx2_GNU2_22_A(__m256d);
            __m256d _ZGVdN4vv_pow_avx2_GNU2_22_A(__m256d, __m256d);
            __m256d _ZGVdN4v_log_avx2_GNU2_22_A(__m256d);
        }

        static inline __m256d vec256_expd(__m256d x) 
        {
            return _ZGVdN4v_exp_avx2_GNU2_22_A(x);
        }
        static inline __m256d vec256_powd(__m256d x, __m256d y) 
        {
            return _ZGVdN4vv_pow_avx2_GNU2_22_A(x, y);
        }
        static inline __m256d vec256_logd(__m256d x) 
        {
            return _ZGVdN4v_log_avx2_GNU2_22_A(x);
        }

    #else

        extern __m256d vec256_expd(__m256d) asm("_ZGVdN4v_exp");
        extern __m256d vec256_powd(__m256d, __m256d) asm("_ZGVdN4vv_pow");
        extern __m256d vec256_logd(__m256d) asm("_ZGVdN4v_log");

    #endif
}
#endif
