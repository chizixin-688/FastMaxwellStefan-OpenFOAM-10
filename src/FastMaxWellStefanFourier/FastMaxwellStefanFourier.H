/*---------------------------------------------------------------------------*\
  This interface file is derived from the OpenFOAM solver framework.

  Original OpenFOAM implementation:
      OpenFOAM Foundation

  Modified by:
      Zixin Chi, 2026

  Purpose of modification:
      1. Removed selected core computational routines from this file.
         These routines are re-implemented in FastMaxwellStefan developed
         by the author.
      2. Provide interface for laminar transport model

  All modifications comply with GNU GPL licensing requirements.

\*---------------------------------------------------------------------------*/

//---------------------------------
// 1. Standard C++ library headers
//---------------------------------
#include <vector>

//---------------------------------
// 2. SIMD headers
//---------------------------------
#include <immintrin.h>

//---------------------------------
// 3. OpenFOAM-10 headers
//---------------------------------
#include "fvMatrix.H"
#include "fvmLaplacian.H"
#include "laminarThermophysicalTransportModel.H"
#include "compressibleMomentumTransportModel.H"
#include "fluidThermo.H"
#include "fluidReactionThermo.H"

//---------------------------------
// 4. FastMaxwellStefan headers
//---------------------------------
#include "FastMaxwellStefan.H"

//=============================================================================//
#ifndef FastMaxwellStefanFourier_H
#define FastMaxwellStefanFourier_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarThermophysicalTransportModels
{

/*---------------------------------------------------------------------------*\
                Class FastMaxwellStefanFourier Declaration
\*---------------------------------------------------------------------------*/

template<class BasicThermophysicalTransportModel>
class FastMaxwellStefanFourier
:
    public laminarThermophysicalTransportModel
    <
        BasicThermophysicalTransportModel
    >
{

    // Private data

            label nSpecies;

            FastMaxwellStefan MS;

public:

    typedef typename BasicThermophysicalTransportModel::alphaField
        alphaField;

    typedef typename
        BasicThermophysicalTransportModel::momentumTransportModel
        momentumTransportModel;

    typedef typename BasicThermophysicalTransportModel::thermoModel
        thermoModel;


    //- Runtime type information
    TypeName("FastMaxwellStefanFourier");

    // Constructors

        //- Construct from a momentum transport model and a thermo model
        FastMaxwellStefanFourier
        (
            const momentumTransportModel& momentumTransport,
            const thermoModel& thermo
        );

    //- Destructor
    virtual ~FastMaxwellStefanFourier();

    // Member Functions

        //- Read thermophysicalTransport dictionary
        virtual bool read();

        //- Effective mass diffusion coefficient
        //  for a given specie mass-fraction [kg/m/s]
        //  This is the self-diffusion coefficient
        virtual tmp<volScalarField> DEff(const volScalarField& Yi) const;

        //- Effective mass diffusion coefficient
        //  for a given specie mass-fraction for patch [kg/m/s]
        virtual tmp<scalarField> DEff
        (
            const volScalarField& Yi,
            const label patchi
        ) const;

        //- Return the heat flux [W/m^2]
        virtual tmp<surfaceScalarField> q() const;

        //- Return the source term for the energy equation
        virtual tmp<fvScalarMatrix> divq(volScalarField& he) const;

        //- Return the specie flux for the given specie mass-fraction [kg/m^2/s]
        virtual tmp<surfaceScalarField> j(const volScalarField& Yi) const;

        //- Return the source term for the given specie mass-fraction equation
        virtual tmp<fvScalarMatrix> divj(volScalarField& Yi) const;

        //- Update the diffusion coefficients and flux corrections
        virtual void correct();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace laminarThermophysicalTransportModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "FastMaxwellStefanFourier.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
