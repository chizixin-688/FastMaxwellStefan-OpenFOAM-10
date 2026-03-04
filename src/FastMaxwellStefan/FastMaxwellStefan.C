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


#include "FastMaxwellStefan.H"

#include <cstring>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
// Construct this class with the thermophysics field by passing reference
// rather than a template parameter
FastMaxwellStefan::FastMaxwellStefan
(   
    const dictionary& dict_,
    const fvMesh& mesh,
    const basicSpecieMixture& mixture,
    const volScalarField& Temperature,
    const volScalarField& Pressure,
    const volScalarField& Density
)
:
    nSpecies(mixture.species().size()),
    logPolynomial_(false),
    soretEffect(dict_.lookupOrDefault("soretEffect", false)),
    Mesh_(mesh),
    Mixture_(mixture),
    Lu(this->nSpecies),
    ckReader(mixture.species(),this->logPolynomial_),
    Mat0(nullptr),
    Mat1(nullptr),
    Mat2(nullptr),
    Arr0(nullptr),
    Arr1(nullptr),
    Arr2(nullptr),
    LogT(nullptr),
    W0511_(nullptr),
    W0489_(nullptr),
    invW_(nullptr),
    calcABPtr(nullptr),
    MMPtr(nullptr),
    T_(Temperature),
    p_(Pressure),
    rho_(Density),
    DFuncs_(this->nSpecies),
    Dij_(this->nSpecies),
    Dii_(this->nSpecies),
    DTi_(this->nSpecies),
    jexp_(this->nSpecies),
    YPtrs(this->nSpecies),
    DijPtrs(this->nSpecies)
{
    // Allocate memory for row pointer
    {
        int times = mixture.Y()[0].internalField().size();
        forAll(mixture.Y()[0].boundaryField(),patchi)
        {
            int nPatchs = mixture.Y()[0].boundaryField()[patchi].size();
            if(times<nPatchs){times=nPatchs;}
        }

        const int size0 = this->nSpecies*this->nSpecies*sizeof(double);
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat0), 32, size0*times)){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat1), 32, size0)){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat2), 32, size0)){throw std::bad_alloc();}
        std::memset(this->Mat0, 0, size0*times);
        std::memset(this->Mat1, 0, size0);
        std::memset(this->Mat2, 0, size0);

        const int size1 = this->nSpecies*sizeof(double);
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr0), 32, size1*times)){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr1), 32, size1)){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr2), 32, size1)){throw std::bad_alloc();}
        std::memset(this->Arr0, 0, size1*times);
        std::memset(this->Arr1, 0, size1);
        std::memset(this->Arr2, 0, size1);

        if(posix_memalign(reinterpret_cast<void**>(&this->LogT), 32, times*sizeof(double))){throw std::bad_alloc();}
        std::memset(this->LogT, 0, times);

        if(posix_memalign(reinterpret_cast<void**>(&this->W0511_), 32, this->nSpecies*sizeof(double))){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->W0489_), 32, this->nSpecies*sizeof(double))){throw std::bad_alloc();}
        if(posix_memalign(reinterpret_cast<void**>(&this->invW_), 32, this->nSpecies*sizeof(double))){throw std::bad_alloc();}
    
        for(int i=0; i<this->nSpecies; i++)
        {
            this->W0511_[i] = std::pow(mixture.Wi(i),0.511);
            this->W0489_[i] = std::pow(mixture.Wi(i),0.489);
            this->invW_[i] = 1.0/mixture.Wi(i);
        }
    }

    // Determine the function for computing AB and matrix multipy
    {
        int remain = mixture.species().size()%8;
        if(remain==0)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB3;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_0;
        }
        else if(remain==1)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB0;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_1;   
        }
        else if(remain==2)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB1;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_2;   
        }
        else if(remain==3)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB2;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_3;   
        }
        else if(remain==4)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB3;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_4;   
        }
        else if(remain==5)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB0;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_5;   
        }
        else if(remain==6)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB1;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_6;   
        }
        else if(remain==7)
        {
            this->calcABPtr = &FastMaxwellStefan::ComputeAB2;
            this->MMPtr = &FastMaxwellStefan::MatrixMultiply484_7;   
        }
    }

    // Get interpolation function if logPolynomial is not used
    if(this->logPolynomial_==false)
    {
        const speciesTable& species = this->Mixture_.species();

        const dictionary& Ddict = dict_.subDict("D");

        // Read the array of specie binary mass diffusion coefficient functions
        forAll(species, i)
        {
            this->DFuncs_[i].setSize(species.size());

            forAll(species, j)
            {
                if (j >= i)
                {
                    const word nameij(species[i] + '-' + species[j]);
                    const word nameji(species[j] + '-' + species[i]);

                    word Dname;

                    if (Ddict.found(nameij) && Ddict.found(nameji))
                    {
                        if (i != j)
                        {
                            WarningInFunction
                                << "Binary mass diffusion coefficients "
                                   "for both " << nameij << " and " << nameji
                                << " provided, using " << nameij << endl;
                        }

                        Dname = nameij;
                    }
                    else if (Ddict.found(nameij))
                    {
                        Dname = nameij;
                    }
                    else if (Ddict.found(nameji))
                    {
                        Dname = nameji;
                    }
                    else
                    {
                        FatalIOErrorInFunction(Ddict)
                            << "Binary mass diffusion coefficients for pair "
                            << nameij << " or " << nameji << " not provided"
                            << exit(FatalIOError);
                    }

                    DFuncs_[i].set
                    (
                        j,
                        Function2<scalar>::New(Dname, Ddict).ptr()
                    );
                }
            }
        }
    }

    //Allocate binary diffusion coefficient field
    this->Dij_.resize(this->nSpecies);

    for(int i=0; i<this->nSpecies; i++)
    {
        Dij_[i].setSize(this->nSpecies);
    }

    for(int i=0; i<this->nSpecies; i++)
    {
        Foam::word Iidx = Foam::word(std::to_string(i));
        for(label j=0; j<this->nSpecies; j++)
        {
            Foam::word Jidx = Foam::word(std::to_string(j));
            if (j > i)
            {
                Dij_[i].set
                (
                    j, 
                    volScalarField::New
                    (
                        "D"+Iidx+Jidx+"_",
                        T_.mesh(),
                        dimensionedScalar(dimArea/dimTime, 0)
                    )
                );
            }
            else if(j == i)
            {}
            else if (j < i)
            {
                Dij_[i].set(j, Dij_[j][i].clone());
            }
        }
    }

    forAll(Dii_,i)
    {
        Dii_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                        "Dii_" + this->Mixture_.Y()[i].name(),
                        T_.mesh().time().timeName(),
                        T_.mesh(),
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                ),
                T_.mesh(),
                dimensionedScalar(dimArea/dimTime, 0)
            )
        );
    }

    //Allocate thermal diffusion coefficient field
    if(this->soretEffect==true)
    {
        this->DTi_.resize(this->nSpecies);
        forAll(DTi_,i)
        {
            DTi_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                            "DTii_" + Mixture_.Y()[i].name(),
                            this->Mesh_.time().timeName(),
                            this->Mesh_,
                            IOobject::NO_READ,
                            IOobject::NO_WRITE
                    ),
                    this->Mesh_,
                    dimensionedScalar(dimArea/dimTime, 0)
                )
            );
        }
    }

    for(int i=0; i<this->nSpecies-1; i++)
    {
        this->jexp_.set
        (
            i,
            surfaceScalarField::New
            (
                "jexpi" + Mixture_.Y()[i].name(),
                this->Mesh_,
                dimensionedScalar(dimensionSet(1, -2, -1, 0, 0), 0)
            )
        );
    }

    // Determine the function of LU decomposition
    {
        int remain = mixture.species().size()%4;
        int align = mixture.species().size() - remain;
        std::vector<int> coins = {56,52,48,44,40,36,32,28,24,20,16,12,8,4};
        for (int i = 0; i < 14; i++) 
        { 
            int n = align / coins[i]; 
            Lu.callTable[i] = n; 
            align = align - n*coins[i]; 
        }
    }

}

// * * * * * * * * * * * * * * * * Destructors  * * * * * * * * * * * * * * //
FastMaxwellStefan::~FastMaxwellStefan()
{
    if(this->Mat0!=nullptr){free(this->Mat0);}
    if(this->Mat1!=nullptr){free(this->Mat1);}
    if(this->Mat2!=nullptr){free(this->Mat2);}
    if(this->Arr0!=nullptr){free(this->Arr0);}
    if(this->Arr1!=nullptr){free(this->Arr1);}
    if(this->Arr2!=nullptr){free(this->Arr2);}
    if(this->LogT!=nullptr){free(this->LogT);}
    if(this->W0511_!=nullptr){free(this->W0511_);}
    if(this->W0489_!=nullptr){free(this->W0489_);}
    if(this->invW_!=nullptr){free(this->invW_);}
}
// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// Ns = nSpecies, times=Max(nCells,nPatches)
void FastMaxwellStefan::allocateMemory(int Ns, int times)
{
    const int size0 = Ns*Ns*sizeof(double);
    if(this->Mat0==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat0), 32, size0*times)){throw std::bad_alloc();}
        std::memset(this->Mat0, 0, size0*times);
    }
    if(this->Mat1==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat1), 32, size0)){throw std::bad_alloc();}
        std::memset(this->Mat1, 0, size0);
    }
    
    if(this->Mat2==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Mat2), 32, size0)){throw std::bad_alloc();}
        std::memset(this->Mat2, 0, size0);
    }

    const int size1 = Ns*sizeof(double);
    if(this->Arr0==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr0), 32, size1*times))
            {throw std::bad_alloc();}
        std::memset(this->Arr0, 0, size1*times);
    }
    if(this->Arr1==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr1), 32, size1))
            {throw std::bad_alloc();}
        std::memset(this->Arr1, 0, size1);
    }
    if(this->Arr2==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->Arr2), 32, size1))
            {throw std::bad_alloc();}
        std::memset(this->Arr2, 0, size1);
    }
    if(this->LogT==nullptr)
    {
        if(posix_memalign(reinterpret_cast<void**>(&this->LogT), 32, times*sizeof(double)))
            {throw std::bad_alloc();}
        std::memset(this->LogT, 0, times);
    }
}



void FastMaxwellStefan::transformDiffusionCoefficient
(
    double* __restrict__ M0,
    double* __restrict__ M1,
    double* __restrict__ M2,
    double* __restrict__ A0,
    double* __restrict__ A1,
    double* __restrict__ A2,
    const double* __restrict__ invW,
    int Ns
)
{

    double* __restrict__ Yref = A0;
    double* __restrict__ X = A1;

    double Wm = 0;
    int remain0 = Ns%4;
    for(int i=0; i<Ns-remain0; i=i+4)
    {
        __m256d Yv = _mm256_loadu_pd(&Yref[i+0]);
        __m256d invWv = _mm256_loadu_pd(&invW[i+0]);
        __m256d Xv = _mm256_mul_pd(Yv,invWv);
        _mm256_storeu_pd(&X[i+0],Xv);
        Wm = Wm + hsum4(Xv);
    }
    if(remain0==1)
    {
        X[Ns-1] = Yref[Ns-1]*invW[Ns-1];
        Wm = Wm + X[Ns-1];
    }
    else if(remain0==2)
    {
        X[Ns-2] = Yref[Ns-2]*invW[Ns-2];
        X[Ns-1] = Yref[Ns-1]*invW[Ns-1];
        Wm = Wm + X[Ns-2] + X[Ns-1];
    }
    else if(remain0==3)
    {
        X[Ns-3] = Yref[Ns-3]*invW[Ns-3];
        X[Ns-2] = Yref[Ns-2]*invW[Ns-2];
        X[Ns-1] = Yref[Ns-1]*invW[Ns-1];
        Wm = Wm + X[Ns-3] + X[Ns-2] + X[Ns-1];        
    }
    Wm = 1/Wm;
    for(int i=0; i<Ns; i++)
    {
        X[i] = X[i]*Wm;
    }

    double* __restrict__ WmByWi = A2;

    for(int i=0; i<Ns-remain0; i=i+4)
    {
        __m256d Wmv = _mm256_set1_pd(Wm);
        __m256d invWv = _mm256_loadu_pd(&invW[i+0]);
        __m256d WmByWiv = _mm256_mul_pd(Wmv,invWv);
        _mm256_storeu_pd(&WmByWi[i+0],WmByWiv);
    }
    if(remain0==1)
    {
        WmByWi[Ns-1] = Wm*invW[Ns-1];
    }
    else if(remain0==2)
    {
        WmByWi[Ns-2] = Wm*invW[Ns-2];
        WmByWi[Ns-1] = Wm*invW[Ns-1];
    }
    else if(remain0==3)
    {
        WmByWi[Ns-3] = Wm*invW[Ns-3];
        WmByWi[Ns-2] = Wm*invW[Ns-2];
        WmByWi[Ns-1] = Wm*invW[Ns-1];     
    }

        const double* __restrict__ DD = M0;
        double* __restrict__ RecDD = M1;
        int remain = Ns*Ns%4;
        int n = Ns*Ns;
        __m256d unity = _mm256_set1_pd(1);

        if(remain==0)
        {
            for(int i=0; i<n;i=i+4)
            {
                __m256d DDv = _mm256_loadu_pd(&DD[i]);
                __m256d RecipDDv = _mm256_div_pd(unity,DDv);
                _mm256_storeu_pd(&RecDD[i],RecipDDv);
            }
        }
        else if(remain==1)
        {
            for(int i=0; i<n-1;i=i+4)
            {
                __m256d DDv = _mm256_loadu_pd(&DD[i]);
                __m256d RecipDDv = _mm256_div_pd(unity,DDv);
                _mm256_storeu_pd(&RecDD[i],RecipDDv);
            }
            {
                int i = n-1;
                RecDD[i] = 1.0/DD[i];
            }
        }
        else if(remain ==2)
        {
            for(int i=0; i<n-2;i=i+4)
            {
                __m256d DDv = _mm256_loadu_pd(&DD[i]);
                __m256d RecipDDv = _mm256_div_pd(unity,DDv);
                _mm256_storeu_pd(&RecDD[i],RecipDDv);
            }
            {
                int i = n-2;
                RecDD[i+0] = 1.0/DD[i+0];
                RecDD[i+1] = 1.0/DD[i+1];
            }
        }
        else
        {
            for(int i=0; i<n-3;i=i+4)
            {
                __m256d DDv = _mm256_loadu_pd(&DD[i]);
                __m256d RecipDDv = _mm256_div_pd(unity,DDv);
                _mm256_storeu_pd(&RecDD[i],RecipDDv);
            }
            {
                int i = n-3;
                RecDD[i+0] = 1.0/DD[i+0];
                RecDD[i+1] = 1.0/DD[i+1];
                RecDD[i+2] = 1.0/DD[i+2];
            }
        }
        double* __restrict__ ReciprocalDDid = A0;
        for(int i = 0; i < Ns;i++)
        {
            ReciprocalDDid[i] = RecDD[i*(Ns)+Ns-1];
        }


    double* __restrict__ A = M0;
    double* __restrict__ B = M2;
    std::memset(A,0,Ns*Ns*sizeof(double));
    std::memset(B,0,Ns*Ns*sizeof(double));
    (this->*calcABPtr)(X,A,B,WmByWi,RecDD,ReciprocalDDid,Wm,Ns);
    Lu.ReAssign(A);
    Lu.Block4LUDecompose();

    double* __restrict__ invA = M1;
    std::memset(invA,0,Ns*Ns*sizeof(double));
    Lu.setInvMatrix(invA);
    Lu.calcInvMatrix(invA);

    double* __restrict__ Dref = M0;
    std::memset(Dref,0,Ns*Ns*sizeof(double));
    (this->*MMPtr)(Dref,invA,B,Ns);
}


void FastMaxwellStefan::ComputeAB0
(
    double* __restrict__ X,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    double* __restrict__ WmByWi,
    double* __restrict__ ReciprocalDD,
    double* __restrict__ ReciprocalDDid,
    double Wm,
    int Ns
)
{
    int d = Ns-1;
    int alignN = Ns;

    double WmByWd = WmByWi[d];
    __m256d WmByWdv = _mm256_set1_pd(WmByWd);

    for(int i = 0; i < d; i=i+4)
    {
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;

        __m256d Xi03v = _mm256_loadu_pd(&X[i+0]);

        __m256d Xi03WmByWdByDDi03dv = _mm256_mul_pd(WmByWdv,Xi03v);
        Xi03WmByWdByDDi03dv = _mm256_mul_pd(_mm256_loadu_pd(&ReciprocalDDid[i+0]),Xi03WmByWdByDDi03dv);
        Xi03WmByWdByDDi03dv = -Xi03WmByWdByDDi03dv;

        __m256d Xi03WmByWdv = _mm256_mul_pd(Xi03v,WmByWdv);
        Xi03WmByWdv = -Xi03WmByWdv;

        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];
        double WmByWi2 = WmByWi[i+2];
        double WmByWi3 = WmByWi[i+3];
        for(int j = 0; j < d; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); //4
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         //5

            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[i*alignN+j+0]); //6
            __m256d Xi0v = broadcast0_256to256(Xi03v);                             //7

            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi03WmByWdByDDi03dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7

            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi03WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6

            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5

            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_256to256(Xi03v);

            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi03WmByWdByDDi03dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);

            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi03WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);

            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);

            __m256d RDDi2j03v = _mm256_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m256d Xi2v = broadcast2_256to256(Xi03v);

            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi03WmByWdByDDi03dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);

            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi03WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);

            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);

            __m256d RDDi3j03v = _mm256_loadu_pd(&ReciprocalDD[(i+3)*alignN+j]);
            __m256d Xi3v = broadcast3_256to256(Xi03v);

            // Compute Ai2j03
            __m256d Ai3j03v = _mm256_mul_pd(WmByWj03v,RDDi3j03v);
            __m256d Xi3WmByWdByDDi2dv = broadcast3_256to256(Xi03WmByWdByDDi03dv);
            Ai3j03v = _mm256_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);

            // Compute Bi2j03
            __m256d Xi3WmByWdv = broadcast3_256to256(Xi03WmByWdv);
            __m256d Bi3j03v = _mm256_fmadd_pd(Xi3v,WmByWj03v,Xi3WmByWdv);
            _mm256_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);

            // Compute Sum0
            __m256d WmByWi3v = _mm256_set1_pd(WmByWi3);
            __m256d sum3v = _mm256_mul_pd(RDDi3j03v,Xj03v);
            sum3v = _mm256_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum4(sum3v);
        }

        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;
        matB[(i+2)*alignN+(i+2)] -= WmByWi2;
        matB[(i+3)*alignN+(i+3)] -= WmByWi3;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
        matA[(i+2)*alignN+(i+2)] -= (sum2 + X[d]*WmByWi[i+2]*(ReciprocalDDid[i+2]));
        matA[(i+3)*alignN+(i+3)] -= (sum3 + X[d]*WmByWi[i+3]*(ReciprocalDDid[i+3]));
    }
}

void FastMaxwellStefan::ComputeAB1
(
    double* __restrict__ X,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    double* __restrict__ WmByWi,
    double* __restrict__ ReciprocalDD,
    double* __restrict__ ReciprocalDDid,
    double Wm,
    int Ns
)
{

    int d = Ns-1;
    int alignN = Ns;
    double WmByWd = WmByWi[d];

    __m256d WmByWdv = _mm256_set1_pd(WmByWd);
    for(int i = 0; i < d-1; i=i+4)
    {
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;

        __m256d Xi03v = _mm256_loadu_pd(&X[i+0]);

        __m256d Xi03WmByWdByDDi03dv = _mm256_mul_pd(WmByWdv,Xi03v);
        Xi03WmByWdByDDi03dv = _mm256_mul_pd(_mm256_loadu_pd(&ReciprocalDDid[i+0]),Xi03WmByWdByDDi03dv);
        Xi03WmByWdByDDi03dv = -Xi03WmByWdByDDi03dv;

        __m256d Xi03WmByWdv = _mm256_mul_pd(Xi03v,WmByWdv);
        Xi03WmByWdv = -Xi03WmByWdv;


        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];
        double WmByWi2 = WmByWi[i+2];
        double WmByWi3 = WmByWi[i+3];
        for(int j = 0; j < d-1; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); 
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         


            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); 
            __m256d Xi0v = broadcast0_256to256(Xi03v);                             
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi03WmByWdByDDi03dv);  
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi03WmByWdv);                 
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              
            sum0 += hsum4(sum0v);                                          


            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_256to256(Xi03v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi03WmByWdByDDi03dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi03WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum1
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);


            __m256d RDDi2j03v = _mm256_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m256d Xi2v = broadcast2_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi03WmByWdByDDi03dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi03WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            // Compute Sum2
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);



            __m256d RDDi3j03v = _mm256_loadu_pd(&ReciprocalDD[(i+3)*alignN+j]);
            __m256d Xi3v = broadcast3_256to256(Xi03v);
            // Compute Ai3j03
            __m256d Ai3j03v = _mm256_mul_pd(WmByWj03v,RDDi3j03v);
            __m256d Xi3WmByWdByDDi2dv = broadcast3_256to256(Xi03WmByWdByDDi03dv);
            Ai3j03v = _mm256_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);
            // Compute Bi3j03
            __m256d Xi3WmByWdv = broadcast3_256to256(Xi03WmByWdv);
            __m256d Bi3j03v = _mm256_fmadd_pd(Xi3v,WmByWj03v,Xi3WmByWdv);
            _mm256_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);
            // Compute Sum3
            __m256d WmByWi3v = _mm256_set1_pd(WmByWi3);
            __m256d sum3v = _mm256_mul_pd(RDDi3j03v,Xj03v);
            sum3v = _mm256_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum4(sum3v);
        }
        {
            int j = d-1;
            double WmByWj0 = WmByWi[j+0]; 
            double Xj0 = X[j+0];         


            double Xi0 = get_elem0(Xi03v);
            double Xi0WmByWdByDDi0d = get_elem0(Xi03WmByWdByDDi03dv);
            matA[(i+0)*alignN+j+0] = Xi0WmByWdByDDi0d + (Xi0*WmByWj0)*(ReciprocalDD[(i+0)*alignN+j]);
            matB[(i+0)*alignN+j+0] = Xi0*WmByWj0 + get_elem0(Xi03WmByWdv);
            sum0 = sum0 + (Xj0*WmByWi0)*(ReciprocalDD[(i+0)*alignN+j]);

            double Xi1 = get_elem1(Xi03v);
            double Xi1WmByWdByDDi1d = get_elem1(Xi03WmByWdByDDi03dv);
            matA[(i+1)*alignN+j+0] = Xi1WmByWdByDDi1d + (Xi1*WmByWj0)*(ReciprocalDD[(i+1)*alignN+j]);
            matB[(i+1)*alignN+j+0] = Xi1*WmByWj0 + get_elem1(Xi03WmByWdv);
            sum1 = sum1 + (Xj0*WmByWi1)*(ReciprocalDD[(i+1)*alignN+j]);

            double Xi2 = get_elem2(Xi03v);
            double Xi2WmByWdByDDi2d = get_elem2(Xi03WmByWdByDDi03dv);
            matA[(i+2)*alignN+j+0] = Xi2WmByWdByDDi2d + (Xi2*WmByWj0)*(ReciprocalDD[(i+2)*alignN+j]);
            matB[(i+2)*alignN+j+0] = Xi2*WmByWj0 + get_elem2(Xi03WmByWdv);
            sum2 = sum2 + (Xj0*WmByWi2)*(ReciprocalDD[(i+2)*alignN+j]);

            double Xi3 = get_elem3(Xi03v);
            double Xi3WmByWdByDDi3d = get_elem3(Xi03WmByWdByDDi03dv);
            matA[(i+3)*alignN+j+0] = Xi3WmByWdByDDi3d + (Xi3*WmByWj0)*(ReciprocalDD[(i+3)*alignN+j]);
            matB[(i+3)*alignN+j+0] = Xi3*WmByWj0 + get_elem3(Xi03WmByWdv);
            sum3 = sum3 + (Xj0*WmByWi3)*(ReciprocalDD[(i+3)*alignN+j]);
        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;
        matB[(i+2)*alignN+(i+2)] -= WmByWi2;
        matB[(i+3)*alignN+(i+3)] -= WmByWi3;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
        matA[(i+2)*alignN+(i+2)] -= (sum2 + X[d]*WmByWi[i+2]*(ReciprocalDDid[i+2]));
        matA[(i+3)*alignN+(i+3)] -= (sum3 + X[d]*WmByWi[i+3]*(ReciprocalDDid[i+3]));
    }

    {
        int i = d-1;
        double sum0 = 0;


        double Xi0 = X[i+0];

        double Xi0WmByWdByDDi0d = -Xi0*WmByWd*ReciprocalDDid[i+0];

        double Xi0WmByWd = -Xi0*WmByWd;

        double WmByWi0 = WmByWi[i+0];

        for(int j = 0; j < d-1; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); 
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         

            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); 
            __m256d Xi0v = _mm256_set1_pd(Xi0);                             
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               
            __m256d Xi0WmByWdByDDi0dv = _mm256_set1_pd(Xi0WmByWdByDDi0d);  
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          
            // Compute Bi0j03
            __m256d Xi0WmByWdv = _mm256_set1_pd(Xi0WmByWd);                 
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              
            sum0 += hsum4(sum0v);                                          
        }
        {
            int j = d-1;

            matA[(i+0)*alignN+j+0] = Xi0WmByWdByDDi0d + (Xi0*WmByWi[j+0])*(ReciprocalDD[(i+0)*alignN+j]);
            matB[(i+0)*alignN+j+0] = Xi0*WmByWi[j+0] + Xi0WmByWd;
            sum0 = sum0 + (X[j+0]*WmByWi0)*(ReciprocalDD[(i+0)*alignN+j]);
        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
    }
}

void FastMaxwellStefan::ComputeAB2
(
    double* __restrict__ X,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    double* __restrict__ WmByWi,
    double* __restrict__ ReciprocalDD,
    double* __restrict__ ReciprocalDDid,
    double Wm,
    int Ns
)
{

    int d = Ns-1;
    int alignN = Ns;
    double WmByWd = WmByWi[d];

    __m256d WmByWdv = _mm256_set1_pd(WmByWd);
    for(int i = 0; i < d-2; i=i+4)
    {
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;

        __m256d Xi03v = _mm256_loadu_pd(&X[i+0]);


        __m256d Xi03WmByWdByDDi03dv = _mm256_mul_pd(WmByWdv,Xi03v);
        Xi03WmByWdByDDi03dv = _mm256_mul_pd(_mm256_loadu_pd(&ReciprocalDDid[i+0]),Xi03WmByWdByDDi03dv);
        Xi03WmByWdByDDi03dv = -Xi03WmByWdByDDi03dv;

        __m256d Xi03WmByWdv = _mm256_mul_pd(Xi03v,WmByWdv);
        Xi03WmByWdv = -Xi03WmByWdv;


        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];
        double WmByWi2 = WmByWi[i+2];
        double WmByWi3 = WmByWi[i+3];
        for(int j = 0; j < d-2; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); //4
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         //5


            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m256d Xi0v = broadcast0_256to256(Xi03v);                             //7
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi03WmByWdByDDi03dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi03WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5



            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_256to256(Xi03v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi03WmByWdByDDi03dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi03WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);



            __m256d RDDi2j03v = _mm256_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m256d Xi2v = broadcast2_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi03WmByWdByDDi03dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi03WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);



            __m256d RDDi3j03v = _mm256_loadu_pd(&ReciprocalDD[(i+3)*alignN+j]);
            __m256d Xi3v = broadcast3_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai3j03v = _mm256_mul_pd(WmByWj03v,RDDi3j03v);
            __m256d Xi3WmByWdByDDi2dv = broadcast3_256to256(Xi03WmByWdByDDi03dv);
            Ai3j03v = _mm256_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);
            // Compute Bi2j03
            __m256d Xi3WmByWdv = broadcast3_256to256(Xi03WmByWdv);
            __m256d Bi3j03v = _mm256_fmadd_pd(Xi3v,WmByWj03v,Xi3WmByWdv);
            _mm256_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);
            // Compute Sum0
            __m256d WmByWi3v = _mm256_set1_pd(WmByWi3);
            __m256d sum3v = _mm256_mul_pd(RDDi3j03v,Xj03v);
            sum3v = _mm256_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum4(sum3v);
        }

        {
            int j = d-2;
            __m128d WmByWj01v = _mm_loadu_pd(&WmByWi[j+0]); //4
            __m128d Xj01v = _mm_loadu_pd(&X[j+0]);         //5


            __m128d RDDi0j01v = _mm_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m128d Xi0v = _mm256_castpd256_pd128(broadcast0_256to256(Xi03v));
            
            // Compute Ai0j03
            __m128d Ai0j03v = _mm_mul_pd(WmByWj01v,RDDi0j01v);               //8
            __m128d Xi0WmByWdByDDi0dv = _mm256_castpd256_pd128(broadcast0_256to256(Xi03WmByWdByDDi03dv));  //9
            Ai0j03v = _mm_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m128d Xi0WmByWdv = _mm256_castpd256_pd128(broadcast0_256to256(Xi03WmByWdv));                 //8
            __m128d Bi0j03v = _mm_fmadd_pd(Xi0v,WmByWj01v,Xi0WmByWdv);       //7
            _mm_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m128d WmByWi0v = _mm_set1_pd(WmByWi0);                         //7
            __m128d sum0v = _mm_mul_pd(RDDi0j01v,Xj01v);                     //7
            sum0v = _mm_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum2(sum0v);                                          //5


            __m128d RDDi1j03v = _mm_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m128d Xi1v = _mm256_castpd256_pd128(broadcast1_256to256(Xi03v));
            // Compute Ai1j03
            __m128d Ai1j03v = _mm_mul_pd(WmByWj01v,RDDi1j03v);
            __m128d Xi1WmByWdByDDi1dv = _mm256_castpd256_pd128(broadcast1_256to256(Xi03WmByWdByDDi03dv));
            Ai1j03v = _mm_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m128d Xi1WmByWdv = _mm256_castpd256_pd128(broadcast1_256to256(Xi03WmByWdv));
            __m128d Bi1j03v = _mm_fmadd_pd(Xi1v,WmByWj01v,Xi1WmByWdv);
            _mm_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m128d WmByWi1v = _mm_set1_pd(WmByWi1);
            __m128d sum1v = _mm_mul_pd(RDDi1j03v,Xj01v);
            sum1v = _mm_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum2(sum1v);



            __m128d RDDi2j03v = _mm_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m128d Xi2v = _mm256_castpd256_pd128(broadcast2_256to256(Xi03v));
            // Compute Ai2j03
            __m128d Ai2j03v = _mm_mul_pd(WmByWj01v,RDDi2j03v);
            __m128d Xi2WmByWdByDDi2dv = _mm256_castpd256_pd128(broadcast2_256to256(Xi03WmByWdByDDi03dv));
            Ai2j03v = _mm_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            // Compute Bi2j03
            __m128d Xi2WmByWdv = _mm256_castpd256_pd128(broadcast2_256to256(Xi03WmByWdv));
            __m128d Bi2j03v = _mm_fmadd_pd(Xi2v,WmByWj01v,Xi2WmByWdv);
            _mm_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            // Compute Sum0
            __m128d WmByWi2v = _mm_set1_pd(WmByWi2);
            __m128d sum2v = _mm_mul_pd(RDDi2j03v,Xj01v);
            sum2v = _mm_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum2(sum2v);



            __m128d RDDi3j03v = _mm_loadu_pd(&ReciprocalDD[(i+3)*alignN+j]);
            __m128d Xi3v = _mm256_castpd256_pd128(broadcast3_256to256(Xi03v));
            // Compute Ai2j03
            __m128d Ai3j03v = _mm_mul_pd(WmByWj01v,RDDi3j03v);
            __m128d Xi3WmByWdByDDi2dv = _mm256_castpd256_pd128(broadcast3_256to256(Xi03WmByWdByDDi03dv));
            Ai3j03v = _mm_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);
            // Compute Bi2j03
            __m128d Xi3WmByWdv = _mm256_castpd256_pd128(broadcast3_256to256(Xi03WmByWdv));
            __m128d Bi3j03v = _mm_fmadd_pd(Xi3v,WmByWj01v,Xi3WmByWdv);
            _mm_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);
            // Compute Sum0
            __m128d WmByWi3v = _mm_set1_pd(WmByWi3);
            __m128d sum3v = _mm_mul_pd(RDDi3j03v,Xj01v);
            sum3v = _mm_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum2(sum3v);
        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;
        matB[(i+2)*alignN+(i+2)] -= WmByWi2;
        matB[(i+3)*alignN+(i+3)] -= WmByWi3;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
        matA[(i+2)*alignN+(i+2)] -= (sum2 + X[d]*WmByWi[i+2]*(ReciprocalDDid[i+2]));
        matA[(i+3)*alignN+(i+3)] -= (sum3 + X[d]*WmByWi[i+3]*(ReciprocalDDid[i+3]));
    }

    {
        int i = d-2;
        double sum0 = 0;
        double sum1 = 0;


        __m128d Xi01v = _mm_loadu_pd(&X[i+0]);


        __m128d Xi01WmByWdByDDi01dv = _mm_mul_pd(_mm256_castpd256_pd128(WmByWdv),Xi01v);
        Xi01WmByWdByDDi01dv = _mm_mul_pd(_mm_loadu_pd(&ReciprocalDDid[i+0]),Xi01WmByWdByDDi01dv);
        Xi01WmByWdByDDi01dv = -Xi01WmByWdByDDi01dv;

        __m128d Xi01WmByWdv = _mm_mul_pd(Xi01v,_mm256_castpd256_pd128(WmByWdv));
        Xi01WmByWdv = -Xi01WmByWdv;


        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];

        for(int j = 0; j < d-2; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); //4
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         //5

            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m256d Xi0v = broadcast0_128to256(Xi01v);   


            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_128to256(Xi01WmByWdByDDi01dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_128to256(Xi01WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5



            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_128to256(Xi01v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_128to256(Xi01WmByWdByDDi01dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_128to256(Xi01WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);
        }

        {
            int j = d-2;
            __m128d WmByWj01v = _mm_loadu_pd(&WmByWi[j+0]); //4
            __m128d Xj01v = _mm_loadu_pd(&X[j+0]);         //5


            __m128d RDDi0j01v = _mm_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m128d Xi0v = (broadcast0_128to128(Xi01v));
            
            // Compute Ai0j03
            __m128d Ai0j03v = _mm_mul_pd(WmByWj01v,RDDi0j01v);               //8
            __m128d Xi0WmByWdByDDi0dv = (broadcast0_128to128(Xi01WmByWdByDDi01dv));  //9
            Ai0j03v = _mm_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m128d Xi0WmByWdv = (broadcast0_128to128(Xi01WmByWdv));                 //8
            __m128d Bi0j03v = _mm_fmadd_pd(Xi0v,WmByWj01v,Xi0WmByWdv);       //7
            _mm_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m128d WmByWi0v = _mm_set1_pd(WmByWi0);                         //7
            __m128d sum0v = _mm_mul_pd(RDDi0j01v,Xj01v);                     //7
            sum0v = _mm_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum2(sum0v);                                          //5

            __m128d RDDi1j03v = _mm_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m128d Xi1v = (broadcast1_128to128(Xi01v));
            // Compute Ai1j03
            __m128d Ai1j03v = _mm_mul_pd(WmByWj01v,RDDi1j03v);
            __m128d Xi1WmByWdByDDi1dv = (broadcast1_128to128(Xi01WmByWdByDDi01dv));
            Ai1j03v = _mm_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m128d Xi1WmByWdv = (broadcast1_128to128(Xi01WmByWdv));
            __m128d Bi1j03v = _mm_fmadd_pd(Xi1v,WmByWj01v,Xi1WmByWdv);
            _mm_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m128d WmByWi1v = _mm_set1_pd(WmByWi1);
            __m128d sum1v = _mm_mul_pd(RDDi1j03v,Xj01v);
            sum1v = _mm_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum2(sum1v);
        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
    }
}

void FastMaxwellStefan::ComputeAB3
(
    double* __restrict__ X,
    double* __restrict__ matA,  
    double* __restrict__ matB,
    double* __restrict__ WmByWi,
    double* __restrict__ ReciprocalDD,
    double* __restrict__ ReciprocalDDid,
    double Wm,
    int Ns
)
{
    int d = Ns-1;
    int alignN = Ns;
    double WmByWd = WmByWi[d];

    __m256d WmByWdv = _mm256_set1_pd(WmByWd);
    for(int i = 0; i < d-3; i=i+4)
    {
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;

        __m256d Xi03v = _mm256_loadu_pd(&X[i+0]);

        __m256d Xi03WmByWdByDDi03dv = _mm256_mul_pd(WmByWdv,Xi03v);
        Xi03WmByWdByDDi03dv = _mm256_mul_pd(_mm256_loadu_pd(&ReciprocalDDid[i+0]),Xi03WmByWdByDDi03dv);
        Xi03WmByWdByDDi03dv = -Xi03WmByWdByDDi03dv;


        __m256d Xi03WmByWdv = _mm256_mul_pd(Xi03v,WmByWdv);
        Xi03WmByWdv = -Xi03WmByWdv;


        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];
        double WmByWi2 = WmByWi[i+2];
        double WmByWi3 = WmByWi[i+3];
        for(int j = 0; j < d-3; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); //4
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         //5

            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m256d Xi0v = broadcast0_256to256(Xi03v);                             //7
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi03WmByWdByDDi03dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi03WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5


            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_256to256(Xi03v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi03WmByWdByDDi03dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi03WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);

            __m256d RDDi2j03v = _mm256_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m256d Xi2v = broadcast2_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi03WmByWdByDDi03dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi03WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);



            __m256d RDDi3j03v = _mm256_loadu_pd(&ReciprocalDD[(i+3)*alignN+j]);
            __m256d Xi3v = broadcast3_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai3j03v = _mm256_mul_pd(WmByWj03v,RDDi3j03v);
            __m256d Xi3WmByWdByDDi2dv = broadcast3_256to256(Xi03WmByWdByDDi03dv);
            Ai3j03v = _mm256_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);
            // Compute Bi2j03
            __m256d Xi3WmByWdv = broadcast3_256to256(Xi03WmByWdv);
            __m256d Bi3j03v = _mm256_fmadd_pd(Xi3v,WmByWj03v,Xi3WmByWdv);
            _mm256_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);
            // Compute Sum0
            __m256d WmByWi3v = _mm256_set1_pd(WmByWi3);
            __m256d sum3v = _mm256_mul_pd(RDDi3j03v,Xj03v);
            sum3v = _mm256_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum4(sum3v);
        }

        {
            int j = d-3;
            __m256d WmByWj03v = _mm256_setr_pd(WmByWi[j+0],WmByWi[j+1],WmByWi[j+2],0); 
            __m256d Xj03v = _mm256_setr_pd(X[j+0],X[j+1],X[j+2],0);        

            __m256d RDDi0j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+0)*alignN+j+0],
                ReciprocalDD[(i+0)*alignN+j+1],
                ReciprocalDD[(i+0)*alignN+j+2],
                0
            ); //6
            __m256d Xi0v = broadcast0_256to256(Xi03v);                             //7
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi03WmByWdByDDi03dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);
            matA[(i+0)*alignN+j+3] = 0;
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi03WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);
            matB[(i+0)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5


            __m256d RDDi1j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+1)*alignN+j+0],
                ReciprocalDD[(i+1)*alignN+j+1],
                ReciprocalDD[(i+1)*alignN+j+2],
                0
            ); //6
            __m256d Xi1v = broadcast1_256to256(Xi03v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi03WmByWdByDDi03dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            matA[(i+1)*alignN+j+3] = 0;
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi03WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            matB[(i+1)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);



            __m256d RDDi2j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+2)*alignN+j+0],
                ReciprocalDD[(i+2)*alignN+j+1],
                ReciprocalDD[(i+2)*alignN+j+2],
                0
            ); //6
            __m256d Xi2v = broadcast2_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi03WmByWdByDDi03dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            matA[(i+2)*alignN+j+3] = 0;
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi03WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            matB[(i+2)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);



            __m256d RDDi3j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+3)*alignN+j+0],
                ReciprocalDD[(i+3)*alignN+j+1],
                ReciprocalDD[(i+3)*alignN+j+2],
                0
            ); //6
            __m256d Xi3v = broadcast3_256to256(Xi03v);
            // Compute Ai2j03
            __m256d Ai3j03v = _mm256_mul_pd(WmByWj03v,RDDi3j03v);
            __m256d Xi3WmByWdByDDi2dv = broadcast3_256to256(Xi03WmByWdByDDi03dv);
            Ai3j03v = _mm256_fmadd_pd(Xi3v,Ai3j03v,Xi3WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+3)*alignN+j+0],Ai3j03v);
            matA[(i+3)*alignN+j+3] = 0;
            // Compute Bi2j03
            __m256d Xi3WmByWdv = broadcast3_256to256(Xi03WmByWdv);
            __m256d Bi3j03v = _mm256_fmadd_pd(Xi3v,WmByWj03v,Xi3WmByWdv);
            _mm256_storeu_pd(&matB[(i+3)*alignN+j+0],Bi3j03v);
            matB[(i+3)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi3v = _mm256_set1_pd(WmByWi3);
            __m256d sum3v = _mm256_mul_pd(RDDi3j03v,Xj03v);
            sum3v = _mm256_mul_pd(WmByWi3v,sum3v);
            sum3 += hsum4(sum3v);
        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;
        matB[(i+2)*alignN+(i+2)] -= WmByWi2;
        matB[(i+3)*alignN+(i+3)] -= WmByWi3;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
        matA[(i+2)*alignN+(i+2)] -= (sum2 + X[d]*WmByWi[i+2]*(ReciprocalDDid[i+2]));
        matA[(i+3)*alignN+(i+3)] -= (sum3 + X[d]*WmByWi[i+3]*(ReciprocalDDid[i+3]));
    }

    {
        int i = d-3;
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;


        __m256d Xi02v = _mm256_setr_pd(X[i+0],X[i+1],X[i+2],0);

        __m256d Xi02WmByWdByDDi02dv = _mm256_mul_pd(WmByWdv,Xi02v);
        Xi02WmByWdByDDi02dv = _mm256_mul_pd(
            _mm256_setr_pd(ReciprocalDDid[i+0],ReciprocalDDid[i+1],ReciprocalDDid[i+2],0
            ),Xi02WmByWdByDDi02dv);
        Xi02WmByWdByDDi02dv = -Xi02WmByWdByDDi02dv;


        __m256d Xi02WmByWdv = _mm256_mul_pd(Xi02v,WmByWdv);
        Xi02WmByWdv = -Xi02WmByWdv;


        double WmByWi0 = WmByWi[i+0];
        double WmByWi1 = WmByWi[i+1];
        double WmByWi2 = WmByWi[i+2];
        for(int j = 0; j < d-3; j=j+4) //4
        {
            __m256d WmByWj03v = _mm256_loadu_pd(&WmByWi[j+0]); //4
            __m256d Xj03v = _mm256_loadu_pd(&X[j+0]);         //5

            __m256d RDDi0j03v = _mm256_loadu_pd(&ReciprocalDD[(i+0)*alignN+j]); //6
            __m256d Xi0v = broadcast0_256to256(Xi02v);                             //7
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi02WmByWdByDDi02dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi02WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5


            __m256d RDDi1j03v = _mm256_loadu_pd(&ReciprocalDD[(i+1)*alignN+j]);
            __m256d Xi1v = broadcast1_256to256(Xi02v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi02WmByWdByDDi02dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi02WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);

            __m256d RDDi2j03v = _mm256_loadu_pd(&ReciprocalDD[(i+2)*alignN+j]);
            __m256d Xi2v = broadcast2_256to256(Xi02v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi02WmByWdByDDi02dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi02WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);
        }

        {
            int j = d-3;
            __m256d WmByWj03v = _mm256_setr_pd(WmByWi[j+0],WmByWi[j+1],WmByWi[j+2],0); 
            __m256d Xj03v = _mm256_setr_pd(X[j+0],X[j+1],X[j+2],0);        

            __m256d RDDi0j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+0)*alignN+j+0],
                ReciprocalDD[(i+0)*alignN+j+1],
                ReciprocalDD[(i+0)*alignN+j+2],
                0
            ); //6
            __m256d Xi0v = broadcast0_256to256(Xi02v);                             //7
            // Compute Ai0j03
            __m256d Ai0j03v = _mm256_mul_pd(WmByWj03v,RDDi0j03v);               //8
            __m256d Xi0WmByWdByDDi0dv = broadcast0_256to256(Xi02WmByWdByDDi02dv);  //9
            Ai0j03v = _mm256_fmadd_pd(Xi0v,Ai0j03v,Xi0WmByWdByDDi0dv);          //8
            _mm256_storeu_pd(&matA[(i+0)*alignN+j+0],Ai0j03v);                          //7
            matA[(i+0)*alignN+j+3] = 0;
            // Compute Bi0j03
            __m256d Xi0WmByWdv = broadcast0_256to256(Xi02WmByWdv);                 //8
            __m256d Bi0j03v = _mm256_fmadd_pd(Xi0v,WmByWj03v,Xi0WmByWdv);       //7
            _mm256_storeu_pd(&matB[(i+0)*alignN+j+0],Bi0j03v);                          //6
            matB[(i+0)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi0v = _mm256_set1_pd(WmByWi0);                         //7
            __m256d sum0v = _mm256_mul_pd(RDDi0j03v,Xj03v);                     //7
            sum0v = _mm256_mul_pd(WmByWi0v,sum0v);                              //6
            sum0 += hsum4(sum0v);                                          //5


            __m256d RDDi1j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+1)*alignN+j+0],
                ReciprocalDD[(i+1)*alignN+j+1],
                ReciprocalDD[(i+1)*alignN+j+2],
                0
            ); //6
            __m256d Xi1v = broadcast1_256to256(Xi02v);
            // Compute Ai1j03
            __m256d Ai1j03v = _mm256_mul_pd(WmByWj03v,RDDi1j03v);
            __m256d Xi1WmByWdByDDi1dv = broadcast1_256to256(Xi02WmByWdByDDi02dv);
            Ai1j03v = _mm256_fmadd_pd(Xi1v,Ai1j03v,Xi1WmByWdByDDi1dv);
            _mm256_storeu_pd(&matA[(i+1)*alignN+j+0],Ai1j03v);
            matA[(i+1)*alignN+j+3] = 0;
            // Compute Bi1j03
            __m256d Xi1WmByWdv = broadcast1_256to256(Xi02WmByWdv);
            __m256d Bi1j03v = _mm256_fmadd_pd(Xi1v,WmByWj03v,Xi1WmByWdv);
            _mm256_storeu_pd(&matB[(i+1)*alignN+j+0],Bi1j03v);
            matB[(i+1)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi1v = _mm256_set1_pd(WmByWi1);
            __m256d sum1v = _mm256_mul_pd(RDDi1j03v,Xj03v);
            sum1v = _mm256_mul_pd(WmByWi1v,sum1v);
            sum1 += hsum4(sum1v);



            __m256d RDDi2j03v = _mm256_setr_pd
            (
                ReciprocalDD[(i+2)*alignN+j+0],
                ReciprocalDD[(i+2)*alignN+j+1],
                ReciprocalDD[(i+2)*alignN+j+2],
                0
            ); //6
            __m256d Xi2v = broadcast2_256to256(Xi02v);
            // Compute Ai2j03
            __m256d Ai2j03v = _mm256_mul_pd(WmByWj03v,RDDi2j03v);
            __m256d Xi2WmByWdByDDi2dv = broadcast2_256to256(Xi02WmByWdByDDi02dv);
            Ai2j03v = _mm256_fmadd_pd(Xi2v,Ai2j03v,Xi2WmByWdByDDi2dv);
            _mm256_storeu_pd(&matA[(i+2)*alignN+j+0],Ai2j03v);
            matA[(i+2)*alignN+j+3] = 0;
            // Compute Bi2j03
            __m256d Xi2WmByWdv = broadcast2_256to256(Xi02WmByWdv);
            __m256d Bi2j03v = _mm256_fmadd_pd(Xi2v,WmByWj03v,Xi2WmByWdv);
            _mm256_storeu_pd(&matB[(i+2)*alignN+j+0],Bi2j03v);
            matB[(i+2)*alignN+j+3] = 0;
            // Compute Sum0
            __m256d WmByWi2v = _mm256_set1_pd(WmByWi2);
            __m256d sum2v = _mm256_mul_pd(RDDi2j03v,Xj03v);
            sum2v = _mm256_mul_pd(WmByWi2v,sum2v);
            sum2 += hsum4(sum2v);

        }
        matB[(i+0)*alignN+(i+0)] -= WmByWi0;
        matB[(i+1)*alignN+(i+1)] -= WmByWi1;
        matB[(i+2)*alignN+(i+2)] -= WmByWi2;

        matA[(i+0)*alignN+(i+0)] -= (sum0 + X[d]*WmByWi[i+0]*(ReciprocalDDid[i+0]));
        matA[(i+1)*alignN+(i+1)] -= (sum1 + X[d]*WmByWi[i+1]*(ReciprocalDDid[i+1]));
        matA[(i+2)*alignN+(i+2)] -= (sum2 + X[d]*WmByWi[i+2]*(ReciprocalDDid[i+2]));
    }
}

void FastMaxwellStefan::transformDFieldsUsingInterpolation
(
    double* __restrict__ invWRef,
    SquareMatrix<scalarField*>& DijPtrsRef,
    List<const scalarField*>& YPtrsRef,
    int Ns,
    int times
)
{
    int timesRemain = times%4;
    int NsRemain = Ns%4;
    int NsNs = Ns*Ns;

    double* __restrict__ DD = this->Mat0;
    for(int i = 0; i < Ns; i++)
    {
        for(int j = 0; j < Ns-NsRemain; j=j+4)
        {
            for(int idx=0; idx<times-timesRemain; idx=idx+4)
            {
                __m256d r0 = load256d(&(*DijPtrsRef[i][j+0])[idx+0]);
                __m256d r1 = load256d(&(*DijPtrsRef[i][j+1])[idx+0]);
                __m256d r2 = load256d(&(*DijPtrsRef[i][j+2])[idx+0]);
                __m256d r3 = load256d(&(*DijPtrsRef[i][j+3])[idx+0]);
                this->transpose4x4_pd(r0,r1,r2,r3);
                store256d(&DD[(idx+0)*NsNs+i*Ns+j+0],r0);
                store256d(&DD[(idx+1)*NsNs+i*Ns+j+0],r1);
                store256d(&DD[(idx+2)*NsNs+i*Ns+j+0],r2);
                store256d(&DD[(idx+3)*NsNs+i*Ns+j+0],r3);
            }
            for(int idx=times-timesRemain; idx<times; idx=idx+1)
            {
                DD[(idx+0)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+0];
                DD[(idx+0)*NsNs+i*Ns+j+1] = (*DijPtrsRef[i][j+1])[idx+0];
                DD[(idx+0)*NsNs+i*Ns+j+2] = (*DijPtrsRef[i][j+2])[idx+0];
                DD[(idx+0)*NsNs+i*Ns+j+3] = (*DijPtrsRef[i][j+3])[idx+0];
            }
        }
        for(int j=Ns-NsRemain; j<Ns; j=j+1)
        {
            for(int idx=0; idx<times-timesRemain; idx=idx+4)
            {
                DD[(idx+0)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+0];
                DD[(idx+1)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+1];
                DD[(idx+2)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+2];
                DD[(idx+3)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+3];
            }
            for(int idx=times-timesRemain; idx<times; idx=idx+1)
            {
                DD[(idx+0)*NsNs+i*Ns+j+0] = (*DijPtrsRef[i][j+0])[idx+0];
            }
        }
    }

    double* __restrict__ Y = this->Arr0;
    for(int j = 0; j < Ns-NsRemain; j=j+4)
    {
        for(int idx=0; idx<times-timesRemain; idx=idx+4)
        {
            __m256d r0 = load256d(&(*YPtrsRef[j+0])[idx+0]);
            __m256d r1 = load256d(&(*YPtrsRef[j+1])[idx+0]);
            __m256d r2 = load256d(&(*YPtrsRef[j+2])[idx+0]);
            __m256d r3 = load256d(&(*YPtrsRef[j+3])[idx+0]);
            this->transpose4x4_pd(r0,r1,r2,r3);
            store256d(&Y[(idx+0)*Ns+j+0],r0);
            store256d(&Y[(idx+1)*Ns+j+0],r0);
            store256d(&Y[(idx+2)*Ns+j+0],r0);
            store256d(&Y[(idx+3)*Ns+j+0],r0);
        }
        for(int idx=times-timesRemain; idx<times; idx=idx+1)
        {
            Y[(idx+0)*Ns+j+0] = (*YPtrsRef[j+0])[idx+0];
            Y[(idx+0)*Ns+j+1] = (*YPtrsRef[j+1])[idx+0];
            Y[(idx+0)*Ns+j+2] = (*YPtrsRef[j+2])[idx+0];
            Y[(idx+0)*Ns+j+3] = (*YPtrsRef[j+3])[idx+0];
        }
    }
    for(int j=Ns-NsRemain; j<Ns; j=j+1)
    {
        for(int idx=0; idx<times-timesRemain; idx=idx+4)
        {
            __m256d r0 = load256d(&(*YPtrsRef[j+0])[idx+0]);
            Y[(idx+0)*Ns+j+0] = this->get_elem0(r0);
            Y[(idx+1)*Ns+j+0] = this->get_elem1(r0);
            Y[(idx+2)*Ns+j+0] = this->get_elem2(r0);
            Y[(idx+3)*Ns+j+0] = this->get_elem3(r0);
        }
        for(int idx=times-timesRemain; idx<times; idx=idx+1)
        {
            Y[(idx+0)*Ns+j+0] = (*YPtrsRef[j+0])[idx+0];
        }
    }

    for(int idx=0; idx<times; idx=idx+1)
    {
        double* __restrict__ M0 = &this->Mat0[NsNs*idx];
        double* __restrict__ A0 = &this->Arr0[Ns*idx];

        this->transformDiffusionCoefficient
        (
            M0,
            this->Mat1,
            this->Mat2,
            A0,
            this->Arr1,
            this->Arr2,
            invWRef,
            Ns
        );
    }


    int speciesRemain = Ns%4;
    int cellRemain = times%4;
    for(int i=0; i<Ns; i++)
    {
        for(int j=0; j<Ns-speciesRemain; j=j+4)
        {
            for(int idx=0; idx<times-cellRemain; idx=idx+4)
            {
                __m256d v0 = load256d(&DD[(idx+0)*NsNs+i*Ns+j]);
                __m256d v1 = load256d(&DD[(idx+1)*NsNs+i*Ns+j]);
                __m256d v2 = load256d(&DD[(idx+2)*NsNs+i*Ns+j]);
                __m256d v3 = load256d(&DD[(idx+3)*NsNs+i*Ns+j]);
                this->transpose4x4_pd(v0,v1,v2,v3);
                store256d(&(*DijPtrsRef[i][j+0])[idx],v0);
                store256d(&(*DijPtrsRef[i][j+1])[idx],v1);
                store256d(&(*DijPtrsRef[i][j+2])[idx],v2);
                store256d(&(*DijPtrsRef[i][j+3])[idx],v3);
            }
            for(int idx = times-cellRemain; idx < times; idx=idx+1)
            {
                (*DijPtrsRef[i][j+0])[idx] = DD[idx*NsNs+i*Ns+j+0];
                (*DijPtrsRef[i][j+1])[idx] = DD[idx*NsNs+i*Ns+j+1];
                (*DijPtrsRef[i][j+2])[idx] = DD[idx*NsNs+i*Ns+j+2];
                (*DijPtrsRef[i][j+3])[idx] = DD[idx*NsNs+i*Ns+j+3];
            }
        }
        for(int j=Ns-speciesRemain; j<Ns; j=j+1)
        {
            for(int idx=0; idx<times; idx=idx+1)
            {
                (*DijPtrsRef[i][j])[idx] = DD[idx*NsNs+i*Ns+j];
            }
        }
    }
}


void FastMaxwellStefan::transformDFieldsUsingPolynomial
(
    const double* __restrict__ invWRef,
    const double* __restrict__ T,
    const double* __restrict__ p,
    SquareMatrix<scalarField*>& DijPtrsRef,
    List<const scalarField*>& YPtrsRef,
    int Ns,
    int times
)
{
    int NsRemain = Ns%4;
    int timesRemain = times%4;
    int NsNs = Ns*Ns;

    this->ckReader.getLogT(T,this->LogT,times);
    double* __restrict__ Y = this->Arr0;
    for(int j = 0; j < Ns-NsRemain; j=j+4)
    {
        for(int idx=0; idx<times-timesRemain; idx=idx+4)
        {
            __m256d r0 = load256d(&(*YPtrsRef[j+0])[idx+0]);
            __m256d r1 = load256d(&(*YPtrsRef[j+1])[idx+0]);
            __m256d r2 = load256d(&(*YPtrsRef[j+2])[idx+0]);
            __m256d r3 = load256d(&(*YPtrsRef[j+3])[idx+0]);
            this->transpose4x4_pd(r0,r1,r2,r3);
            store256d(&Y[(idx+0)*Ns+j+0],r0);
            store256d(&Y[(idx+1)*Ns+j+0],r0);
            store256d(&Y[(idx+2)*Ns+j+0],r0);
            store256d(&Y[(idx+3)*Ns+j+0],r0);
        }
        for(int idx=times-timesRemain; idx<times; idx=idx+1)
        {
            Y[(idx+0)*Ns+j+0] = (*YPtrsRef[j+0])[idx+0];
            Y[(idx+0)*Ns+j+1] = (*YPtrsRef[j+1])[idx+0];
            Y[(idx+0)*Ns+j+2] = (*YPtrsRef[j+2])[idx+0];
            Y[(idx+0)*Ns+j+3] = (*YPtrsRef[j+3])[idx+0];
        }
    }
    for(int j=Ns-NsRemain; j<Ns; j=j+1)
    {
        for(int idx=0; idx<times-timesRemain; idx=idx+4)
        {
            __m256d r0 = load256d(&(*YPtrsRef[j+0])[idx+0]);
            Y[(idx+0)*Ns+j+0] = this->get_elem0(r0);
            Y[(idx+1)*Ns+j+0] = this->get_elem1(r0);
            Y[(idx+2)*Ns+j+0] = this->get_elem2(r0);
            Y[(idx+3)*Ns+j+0] = this->get_elem3(r0);
        }
        for(int idx=times-timesRemain; idx<times; idx=idx+1)
        {
            Y[(idx+0)*Ns+j+0] = (*YPtrsRef[j+0])[idx+0];
        }
    }

    for(int idx=0; idx<times; idx=idx+1)
    {
        double* __restrict__ M0 = &this->Mat0[NsNs*idx];
        double* __restrict__ A0 = &this->Arr0[Ns*idx];
        this->ckReader.getDD(M0,LogT[idx],p[idx],Ns);

        this->transformDiffusionCoefficient
        (
            M0,
            this->Mat1,
            this->Mat2,
            A0,
            this->Arr1,
            this->Arr2,
            invWRef,
            Ns
        );
    }

    const double* __restrict__ DRef = this->Mat0;
    int begin = 0;
    int speciesRemain = Ns%4;
    int cellRemain = times%4;
    for(int i = 0; i < Ns; i++)
    {
        for(int j = 0; j < Ns -speciesRemain; j=j+4)
        {
            begin = 0;
            for(int iCell = 0; iCell < times-cellRemain; iCell=iCell+4)
            {
                __m256d v0 = load256d(&DRef[begin+i*Ns+j]);
                begin = begin + Ns*Ns;
                __m256d v1 = load256d(&DRef[begin+i*Ns+j]);
                begin = begin + Ns*Ns;
                __m256d v2 = load256d(&DRef[begin+i*Ns+j]);
                begin = begin + Ns*Ns;
                __m256d v3 = load256d(&DRef[begin+i*Ns+j]);
                begin = begin + Ns*Ns;
                this->transpose4x4_pd(v0,v1,v2,v3);
                store256d(&(*DijPtrsRef[i][j+0])[iCell],v0);
                store256d(&(*DijPtrsRef[i][j+1])[iCell],v1);
                store256d(&(*DijPtrsRef[i][j+2])[iCell],v2);
                store256d(&(*DijPtrsRef[i][j+3])[iCell],v3);
            }
            for(int iCell = times-cellRemain; iCell < times; iCell=iCell+1)
            {
                (*DijPtrsRef[i][j+0])[iCell] = DRef[begin+i*Ns+j+0];
                (*DijPtrsRef[i][j+1])[iCell] = DRef[begin+i*Ns+j+1];
                (*DijPtrsRef[i][j+2])[iCell] = DRef[begin+i*Ns+j+2];
                (*DijPtrsRef[i][j+3])[iCell] = DRef[begin+i*Ns+j+3];
                begin = begin + Ns*Ns;
            }
        }
        for(int j = Ns -speciesRemain; j < Ns ; j=j+1)
        {
            begin = 0;
            for(int iCell = 0; iCell < times; iCell=iCell+1)
            {
                (*DijPtrsRef[i][j])[iCell] = DRef[begin+i*Ns+j];
                begin = begin + Ns*Ns;
            }
        }
    }
}



void FastMaxwellStefan::correctUsingPolynomial()
{
    auto start = std::chrono::high_resolution_clock::now();

    const PtrList<volScalarField>& Yfields = this->Mixture_.Y();
    for(label i=0; i<this->nSpecies-1; i++) {jexp_[i] = Zero;}

    const volScalarField& Y0 = Yfields[0];

    for(label i=0; i<this->nSpecies; i++)
    {
        YPtrs[i] = &Yfields[i].primitiveField();

        DijPtrs[i][i] = &Dii_[i].primitiveFieldRef();

        for(label j=0; j<this->nSpecies; j++)
        {
            if (j != i)
            {
                DijPtrs[i][j] = &Dij_[i][j].primitiveFieldRef();
            }
        }
    }

    this->transformDFieldsUsingPolynomial
    (
        this->invW_,
        &T_.internalField()[0],
        &p_.internalField()[0],
        this->DijPtrs,
        this->YPtrs,
        this->nSpecies,
        T_.internalField().size()
    );

    // Do the boundaryField
    forAll(Y0.boundaryField(), patchi)
    {
        for(label i = 0; i < this->nSpecies;i++)
        {
            YPtrs[i] = &Yfields[i].boundaryField()[patchi];

            DijPtrs[i][i] = &Dii_[i].boundaryFieldRef()[patchi];

            for(label j = 0; j<this->nSpecies; j++)
            {
                if (j != i)
                {
                    DijPtrs[i][j] = &Dij_[i][j].boundaryFieldRef()[patchi];
                }
            }
        }

        // Transform binary mass diffusion coefficients patch field DijPtrs ->
        // generalised Fick's law diffusion coefficients DijPtrs

        this->transformDFieldsUsingPolynomial
        (
            this->invW_,
            &T_.boundaryField()[patchi][0],
            &p_.boundaryField()[patchi][0],
            this->DijPtrs,
            this->YPtrs,
            this->nSpecies,
            T_.boundaryField()[patchi].size()
        );
    }

    // Accumulate the explicit part of the specie mass flux fields
    for(label j=0; j<this->nSpecies-1; j++)
    {
        const surfaceScalarField snGradYj(fvc::snGrad(Yfields[j]));

        for(label i=0; i<this->nSpecies-1; i++)
        {
            if (i != j)
            {
                this->jexp_[i] -= fvc::interpolate(this->rho_*this->Dij_[i][j])*snGradYj;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Info << "Execution time: " 
              << duration.count() << " microseconds" << endl;
}

void FastMaxwellStefan::correctUsingInterpolation()
{
    auto start = std::chrono::high_resolution_clock::now();

    const PtrList<volScalarField>& Yfields = this->Mixture_.Y();

    for(label i=0; i<this->nSpecies-1; i++){jexp_[i] = Zero;}

    for(label i=0; i<this->nSpecies; i++)
    {
        evaluate(Dii_[i],DFuncs_[i][i],p_,T_);

        for(label j=0; j<this->nSpecies; j++)
        {
            if      (j>i)   {this->Dij_[i].set(j, evaluate(DFuncs_[i][j], dimViscosity, p_, T_));}
            else if (j<i)   {this->Dij_[i].set(j, this->Dij_[j][i].clone());}
        }
    }

    //- Transform the binary mass diffusion coefficients into the
    //  the generalised Fick's law diffusion coefficients
    for(label i=0; i<this->nSpecies; i++)
    {
        YPtrs[i] = &Yfields[i].primitiveField();

        DijPtrs[i][i] = &Dii_[i].primitiveFieldRef();

        for(label j=0; j<this->nSpecies; j++)
        {
            if (j != i)
            {
                DijPtrs[i][j] = &this->Dij_[i][j].primitiveFieldRef();
            }
        }
    }

    // Transform binary mass diffusion coefficients internal field DijPtrs ->
    // generalised Fick's law diffusion coefficients DijPtrs
    this->transformDFieldsUsingInterpolation
    (
        this->invW_,
        this->DijPtrs,
        this->YPtrs,
        this->nSpecies,
        this->T_.internalField().size()
    );

    // Do the boundaryField
    forAll(T_.boundaryField(), patchi)
    {

        for(label i = 0; i < this->nSpecies;i++)
        {
            this->YPtrs[i] = &Yfields[i].boundaryField()[patchi];
            this->DijPtrs[i][i] = &Dii_[i].boundaryFieldRef()[patchi];

            for(label j = 0; j<this->nSpecies; j++)
            {
                if (j != i)
                {
                    this->DijPtrs[i][j] = &this->Dij_[i][j].boundaryFieldRef()[patchi];
                }
            }
        }

        this->transformDFieldsUsingInterpolation
        (
            this->invW_,
            this->DijPtrs,
            this->YPtrs,
            this->nSpecies,
            this->T_.boundaryField()[patchi].size()
        );
    }

    // Accumulate the explicit part of the specie mass flux fields
    for(label j=0; j<this->nSpecies-1; j++)
    {
        const surfaceScalarField snGradYj(fvc::snGrad(Yfields[j]));

        for(label i=0; i<this->nSpecies-1; i++)
        {
            if (i != j)
            {
                this->jexp_[i] -= fvc::interpolate(this->rho_*this->Dij_[i][j])*snGradYj;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Info << "Execution time: " 
              << duration.count() << " microseconds" << endl;
}



void FastMaxwellStefan::getSoretCoeffUsingBartlett
(
    const volScalarField& T,
    const volScalarField& p,
    const volScalarField& rho,
    const double* __restrict__ W0511,
    const double* __restrict__ W0489,
    const double* __restrict__ invW,
    const PtrList<volScalarField>& Yfields,
    PtrList<volScalarField>& DT,
    const int Ns
)
{
    const double Ru = 8314.470066505449722171761095523834228515625; 

    // Do the internal field
    const int times0 = T.internalField().size();
    std::vector<double> SumX0511(times0,0);
    std::vector<double> SumX0489(times0,0);
    std::vector<double> r0(times0);
    std::vector<double> Tr0(times0);
    for(int j=0; j<Ns; j++)
    {
        const double W0511j = W0511[j];
        const double W0489j = W0489[j];
        const double invWj = invW[j];

        double* __restrict__ X = &this->Arr0[j*times0];
        double* __restrict__ WX0 = &this->Mat0[j*times0];//5.11
        const double* __restrict__ YPtr = &Yfields[j].internalField()[0];
        const double* __restrict__ rhoPtr = &rho.internalField()[0];
        const double* __restrict__ pPtr = &p.internalField()[0];
        const double* __restrict__ TPtr = &T.internalField()[0];

        for(int idx=0; idx<times0; idx++)
        {
            const double rhoi = rhoPtr[idx];
            const double pi = pPtr[idx];
            const double Yji = YPtr[idx];
            const double Ti = TPtr[idx];
            const double Xji = Yji*rhoi*Ru*Ti/pi*invWj;
            X[idx] = Xji;
            WX0[idx] = W0511j*Xji;
            SumX0511[idx] += W0511j*Xji;
            SumX0489[idx] += W0489j*Xji;
        }
    }
    for(int idx=0; idx<times0; idx++)
    {
        r0[idx] = SumX0511[idx]/SumX0489[idx];
    }
    for(int i=0; i<Ns; i++)
    {
        double* __restrict__ WX0 = &this->Mat0[i*times0];//5.11
        const double* __restrict__ Y = &Yfields[i].internalField()[0];
        double* __restrict__ ptr = &DT[i].primitiveFieldRef()[0];
        for(int idx=0; idx<times0; idx++)
        {
            const double Ti = T.internalField()[idx];
            const double Tr0_ = std::pow(Ti,0.659);
            const double Yiidx = Y[idx];

            const double number = -(2.59e-7);

            ptr[idx] = number*Tr0_*(WX0[idx]/SumX0511[idx]-Yiidx)*(SumX0511[idx]/SumX0489[idx]);
        }
    }

    //Do the boundary field

    for(int patchi=0; patchi<T.boundaryField().size(); patchi++)
    {
        int times1 = T.boundaryField()[patchi].size();
        for(int j=0; j<Ns; j++)
        {
            const double W0511j = W0511[j];
            const double W0489j = W0489[j];
            const double invWj = invW[j];

            double* __restrict__ X = &this->Arr0[j*times1];
            double* __restrict__ WX0 = &this->Mat0[j*times1];//5.11
            const double* __restrict__ YPtr = &Yfields[j].boundaryField()[patchi][0];
            const double* __restrict__ rhoPtr = &rho.boundaryField()[patchi][0];
            const double* __restrict__ pPtr = &p.boundaryField()[patchi][0];
            const double* __restrict__ TPtr = &T.boundaryField()[patchi][0];

            for(int idx=0; idx<times1; idx++)
            {
                const double rhoi = rhoPtr[idx];
                const double pi = pPtr[idx];
                const double Yji = YPtr[idx];
                const double Ti = TPtr[idx];
                const double Xji = Yji*rhoi*Ru*Ti/pi*invWj;
                X[idx] = Xji;
                WX0[idx] = W0511j*Xji;
                SumX0511[idx] += W0511j*Xji;
                SumX0489[idx] += W0489j*Xji;
            }
        }
        for(int idx=0; idx<times1; idx++)
        {
            r0[idx] = SumX0511[idx]/SumX0489[idx];
        }
        for(int i=0; i<Ns; i++)
        {
            double* __restrict__ WX0 = &this->Mat0[i*times1];//5.11
            const double* __restrict__ YPtr = &Yfields[i].boundaryField()[patchi][0];
            const double* __restrict__ TPtr = &T.boundaryField()[patchi][0];
            double* __restrict__ DTPtr = &DT[i].boundaryFieldRef()[patchi][0];

            for(int idx=0; idx<times1; idx++)
            {
                const double Ti = TPtr[idx];
                const double Tr0_ = std::pow(Ti,0.659);
                const double Yiidx = YPtr[idx];

                const double number = -(2.59e-7);

                DTPtr[idx] = number*Tr0_*(WX0[idx]/SumX0511[idx]-Yiidx)*(SumX0511[idx]/SumX0489[idx]);
            }
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
