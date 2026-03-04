/*---------------------------------------------------------------------------*\
  Copyright (C) 2026 Zixin Chi
  All rights reserved.

  This file is an original work developed independently by Zixin Chi.

  This program is distributed under the terms of the
  GNU General Public License v3.0.

  Description
      This file is used to compute generalized Fick diffusion coefficient
      using binary diffusion coefficient.

  Author
      Zixin Chi <chizixin@buaa.edu.cn>

\*---------------------------------------------------------------------------*/


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

inline double FastMaxwellStefan::hsum4(__m256d v) 
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v,1);
    lo = _mm_add_pd(lo,hi);               
    __m128d sh = _mm_unpackhi_pd(lo,lo);  
    __m128d s = _mm_add_sd(lo, sh);       
    return _mm_cvtsd_f64(s);
}

inline __m256d FastMaxwellStefan::hsum4x4(__m256d v0, __m256d v1,__m256d v2, __m256d v3) 
{

    __m128d lo0 = _mm256_castpd256_pd128(v0);   //[v00 v01]
    __m128d hi0 = _mm256_extractf128_pd(v0,1);  //[v02 v03]
    __m128d s0 = _mm_add_pd(lo0, hi0);          //[v00+v02 v01+v03]

    __m128d sh0 = _mm_unpackhi_pd(s0, s0);      //[v01+v03 v01+v03]
    __m128d sum0 = _mm_add_sd(s0, sh0);         //[v00+v01+v02+v03 v01+v01+v03+v03]

    __m128d lo1 = _mm256_castpd256_pd128(v1);
    __m128d hi1 = _mm256_extractf128_pd(v1,1);
    __m128d s1 = _mm_add_pd(lo1, hi1);

    __m128d sh1 = _mm_unpackhi_pd(s1, s1);
    __m128d sum1 = _mm_add_sd(s1, sh1);


    __m128d lo2 = _mm256_castpd256_pd128(v2);
    __m128d hi2 = _mm256_extractf128_pd(v2,1);
    __m128d s2 = _mm_add_pd(lo2, hi2);

    __m128d sh2 = _mm_unpackhi_pd(s2, s2);
    __m128d sum2 = _mm_add_sd(s2, sh2);

    __m128d lo3 = _mm256_castpd256_pd128(v3);
    __m128d hi3 = _mm256_extractf128_pd(v3,1);
    __m128d s3 = _mm_add_pd(lo3, hi3);

    __m128d sh3 = _mm_unpackhi_pd(s3, s3);
    __m128d sum3 = _mm_add_sd(s3, sh3);



    __m128d low  = _mm_unpacklo_pd(sum0, sum1); // [sum0, sum1]
    __m128d high = _mm_unpacklo_pd(sum2, sum3); // [sum2, sum3]

    return _mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 1);
}

inline double FastMaxwellStefan::hsum2(__m128d v) 
{
    __m128d hi = _mm_unpackhi_pd(v, v); 
    __m128d sum = _mm_add_sd(v, hi);    
    double result = _mm_cvtsd_f64(sum);
    return result;
}

inline double FastMaxwellStefan::get_elem0(__m256d vec) 
{
    return _mm256_cvtsd_f64(vec); 
}

inline double FastMaxwellStefan::get_elem0(__m128d vec) 
{
    return _mm_cvtsd_f64(vec); 
}

inline double FastMaxwellStefan::get_elem1(__m256d vec) 
{
    __m128d low = _mm256_castpd256_pd128(vec); 
    return _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
}

inline double FastMaxwellStefan::get_elem1(__m128d vec) 
{
    return _mm_cvtsd_f64(_mm_unpackhi_pd(vec,vec)); 
}

inline double FastMaxwellStefan::get_elem2(__m256d vec) 
{
    __m128d high = _mm256_extractf128_pd(vec, 1); 
    return _mm_cvtsd_f64(high);
}

inline double FastMaxwellStefan::get_elem3(__m256d vec) 
{
    __m128d high = _mm256_extractf128_pd(vec, 1); 
    return _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
}

inline __m256d FastMaxwellStefan::broadcast0_256to256(__m256d vec) 
{
    return _mm256_permute4x64_pd(vec, 0x00); 
}

inline __m256d FastMaxwellStefan::broadcast1_256to256(__m256d vec) 
{
    return _mm256_permute4x64_pd(vec, 0x55); 
}

inline __m256d FastMaxwellStefan::broadcast2_256to256(__m256d vec) 
{
    return _mm256_permute4x64_pd(vec, 0xAA); 
}

inline __m256d FastMaxwellStefan::broadcast3_256to256(__m256d vec) 
{
    return _mm256_permute4x64_pd(vec, 0xFF); 
}

inline __m128d FastMaxwellStefan::broadcast0_128to128(__m128d vec) 
{
    return _mm_shuffle_pd(vec, vec, 0x00);
}

inline __m128d FastMaxwellStefan::broadcast1_128to128(__m128d vec) 
{
    return _mm_shuffle_pd(vec, vec, 0x03);
}

inline __m256d FastMaxwellStefan::broadcast0_128to256(__m128d vec) 
{
    __m128d t = _mm_shuffle_pd(vec, vec, 0x00);
    __m256d x = _mm256_castpd128_pd256(t);    
    return _mm256_permute4x64_pd(x, 0x00);   
}

inline __m256d FastMaxwellStefan::broadcast1_128to256(__m128d v) 
{
    __m128d t = _mm_shuffle_pd(v, v, 0x03);
    __m256d x = _mm256_castpd128_pd256(t);    
    return _mm256_permute4x64_pd(x, 0x00);   
}

inline void FastMaxwellStefan::transpose4x4_pd(__m256d& v0,__m256d& v1,__m256d& v2,__m256d& v3)
{
    // v0 = [a0 a1 a2 a3]
    // v1 = [b0 b1 b2 b3]
    // v2 = [c0 c1 c2 c3]
    // v3 = [d0 d1 d2 d3]
    __m256d t0 = _mm256_unpacklo_pd(v0, v1); // a0 b0 a1 b1
    __m256d t1 = _mm256_unpackhi_pd(v0, v1); // a2 b2 a3 b3
    __m256d t2 = _mm256_unpacklo_pd(v2, v3); // c0 d0 c1 d1
    __m256d t3 = _mm256_unpackhi_pd(v2, v3); // c2 d2 c3 d3

    v0 = _mm256_permute2f128_pd(t0, t2, 0x20); // v0 = [a0 b0 c0 d0]
    v1 = _mm256_permute2f128_pd(t1, t3, 0x20); // v1 = [a1 b1 c1 d1]
    v2 = _mm256_permute2f128_pd(t0, t2, 0x31); // v2 = [a2 b2 c2 d2]
    v3 = _mm256_permute2f128_pd(t1, t3, 0x31); // v3 = [a3 b3 c3 d3]
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
