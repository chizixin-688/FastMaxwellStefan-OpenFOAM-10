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

//---------------------------------
// 1. Standard C++ library headers
//---------------------------------
#include <iostream>
#include <vector>
#include <chrono>

//---------------------------------
// 2. SIMD headers
//---------------------------------
#include <immintrin.h>

//---------------------------------
// 3. OpenFOAM-10 headers
//---------------------------------
#include "SquareMatrix.H"
#include "scalarField.H"
#include "List.H"
#include "geometricOneField.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "speciesTable.H"
#include "basicSpecieMixture.H"
#include "Function2.H"
#include "fvcSnGrad.H"
#include "surfaceInterpolate.H"
#include "Function2Evaluate.H"

//---------------------------------
// 4. FastMaxwellStefan headers
//---------------------------------
#include "LUsolver.H"
#include "CKDDreader.H"

//=============================================================================//
#ifndef FastMaxwellStefan_H
#define FastMaxwellStefan_H

namespace Foam
{

/*---------------------------------------------------------------------------*\
                Class FastMaxwellStefan Declaration
\*---------------------------------------------------------------------------*/

class FastMaxwellStefan
{
        // Total number of species
        int nSpecies;

        // Whether to use log polynomial for binary diffusion coefficient
        bool logPolynomial_;

        // Whether to consider soret effect using Bartlett model
        bool soretEffect;

        // Reference of fvMesh class, used for field management
        const fvMesh& Mesh_;

        // Reference of basicSpecieMixture class, used to obtain species information
        const basicSpecieMixture& Mixture_;

        // A class for LU decomposition, LU backsubstitution
        LUsolver Lu;

        // A class for read CHEMKIN fitting coefficient file and provide DD computation
        CKDDreader ckReader;


        // Array

            double* Mat0;

            double* Mat1;

            double* Mat2;

            double* Arr0;

            double* Arr1;

            double* Arr2;

            double* LogT;

            double* W0511_;
            
            double* W0489_;

            double* invW_;

        // A function pointer for calculation of A and B.
        void (FastMaxwellStefan::*calcABPtr)
        (                
            double* __restrict__ X,
            double* __restrict__ matA,  
            double* __restrict__ matB,
            double* __restrict__ WmByWi,
            double* __restrict__ ReciprocalDD,
            double* __restrict__ ReciprocalDDid,
            double Wm,
            int nSpecies
        )=nullptr;

        // A function pointer for matrix multiplication
        void (FastMaxwellStefan::*MMPtr)
        (
            double* __restrict__ C, 
            double* __restrict__ A, 
            double* __restrict__ B, 
            int N
        )=nullptr;

        // Reference of temperature field
        const volScalarField& T_;

        // Reference of pressure field
        const volScalarField& p_;

        // Reference of density field
        const volScalarField& rho_;

        //- Array of specie binary mass diffusion coefficient functions  [m^2/s]
        List<PtrList<Function2<scalar>>> DFuncs_;

        //- Generalised Fick's law diffusion coefficients field list.
        List<PtrList<volScalarField>> Dij_;

        //- Generalised Fick's law diffusion coefficients field list, diagonal element.
        PtrList<volScalarField> Dii_;

        //- Thermal diffusion coefficients field list.
        PtrList<volScalarField> DTi_;

        //- List of fields of the explicit part of the mass diffusion flux of the species
        PtrList<surfaceScalarField> jexp_;

        //- List of mass-fraction field pointers for the internal mesh or a patch
        List<const scalarField*> YPtrs;

        //- Matrix of mass diffusion coefficient field pointers  for the internal mesh or a patch
        SquareMatrix<scalarField*> DijPtrs;
            
    // Private member functions

        //- Matrix Multiplication for Square Matrix

            void MatrixMultiply484_0
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );        
            void MatrixMultiply484_1
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_2
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_3
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_4
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_5
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_6
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );
            void MatrixMultiply484_7
            (
                double* __restrict__ C, 
                double* __restrict__ A, 
                double* __restrict__ B, 
                int N
            );

        //- Compute A and B matrix

            void ComputeAB0
            (
                double* __restrict__ X,
                double* __restrict__ matA,  
                double* __restrict__ matB,
                double* __restrict__ WmByWi,
                double* __restrict__ ReciprocalDD,
                double* __restrict__ ReciprocalDDid,
                double Wm,
                int nSpecies
            );

            void ComputeAB1
            (
                double* __restrict__ X,
                double* __restrict__ matA,  
                double* __restrict__ matB,
                double* __restrict__ WmByWi,
                double* __restrict__ ReciprocalDD,
                double* __restrict__ ReciprocalDDid,
                double Wm,
                int nSpecies
            );
            void ComputeAB2
            (
                double* __restrict__ X,
                double* __restrict__ matA,  
                double* __restrict__ matB,
                double* __restrict__ WmByWi,
                double* __restrict__ ReciprocalDD,
                double* __restrict__ ReciprocalDDid,
                double Wm,
                int nSpecies
            );
            void ComputeAB3
            (
                double* __restrict__ X,
                double* __restrict__ matA,  
                double* __restrict__ matB,
                double* __restrict__ WmByWi,
                double* __restrict__ ReciprocalDD,
                double* __restrict__ ReciprocalDDid,
                double Wm,
                int nSpecies
            );


        // SIMD inline function

            inline double hsum4(__m256d v);
            inline double hsum2(__m128d v);
            inline __m256d hsum4x4(__m256d A, __m256d B, __m256d C, __m256d D);
            inline double get_elem0(__m256d vec);
            inline double get_elem0(__m128d vec);
            inline double get_elem1(__m256d vec);
            inline double get_elem1(__m128d vec);
            inline double get_elem2(__m256d vec);
            inline double get_elem3(__m256d vec);
            inline __m256d broadcast0_256to256(__m256d v);
            inline __m128d broadcast0_128to128(__m128d v);
            inline __m256d broadcast1_256to256(__m256d v);
            inline __m128d broadcast1_128to128(__m128d v);
            inline __m256d broadcast2_256to256(__m256d v);
            inline __m256d broadcast3_256to256(__m256d v);
            inline __m256d broadcast0_128to256(__m128d v);
            inline __m256d broadcast1_128to256(__m128d v);
            inline void transpose4x4_pd(__m256d& v0,__m256d& v1,__m256d& v2,__m256d& v3);

public:


    // Constructors

        FastMaxwellStefan
        (
            const dictionary& dict_,
            const fvMesh& mesh,
            const basicSpecieMixture& mixture,
            const volScalarField& Temperature,
            const volScalarField& Pressure,
            const volScalarField& Density
        );

        FastMaxwellStefan()=delete;

    //- Destructor
    virtual ~FastMaxwellStefan();

    // Access Functions

        const PtrList<volScalarField>& Dii() const  {return this->Dii_;}

        const PtrList<volScalarField>& DTi() const  {return this->DTi_;}

        const PtrList<surfaceScalarField>& jexp() const  {return this->jexp_;}

        const List<PtrList<volScalarField>>& Dij() const  {return this->Dij_;}

        const bool& logPolynomial()const {return this->logPolynomial_;}

    // Member Functions

        void allocateMemory(int nSpecies, int times);

        void transformDiffusionCoefficient
        (
            double* __restrict__ M0,
            double* __restrict__ M1,
            double* __restrict__ M2,
            double* __restrict__ A0,
            double* __restrict__ A1,
            double* __restrict__ A2,
            const double* __restrict__ invW,
            int nSpecies
        );

        void correctUsingPolynomial();

        void correctUsingInterpolation();

        void transformDFieldsUsingInterpolation
        (
            double* __restrict__ invWRef,
            SquareMatrix<scalarField*>& DijPtrsRef,
            List<const scalarField*>& YPtrsRef,
            int nSpecies,
            int times
        );

        void transformDFieldsUsingPolynomial
        (
            const double* __restrict__ invWRef,
            const double* __restrict__ T,
            const double* __restrict__ p,
            SquareMatrix<scalarField*>& DijPtrsRef,
            List<const scalarField*>& YPtrsRef,
            int nSpecies,
            int times
        );

        void getSoretCoeffUsingBartlett
        (
            const volScalarField& T,
            const volScalarField& p,
            const volScalarField& rho,
            const double* __restrict__ W0511,
            const double* __restrict__ W0489,
            const double* __restrict__ invW,
            const PtrList<volScalarField>& Yfields,
            PtrList<volScalarField>& DT,
            const int nSpecies
        );
        void printMatrix(const double* Mat,int N)
        {
            int size = N;
            for(int i = 0; i < size;i++)
            {
                for(int j = 0; j < size-1;j++)
                {
                    std::cout<<Mat[i*N+j]<<" ";
                }
                int j = size-1;
                std::cout<<Mat[i*N+j]<<std::endl;
            }
        }

        void printDField();
        void printDField(int i);

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#include "FastMaxwellStefanI.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
