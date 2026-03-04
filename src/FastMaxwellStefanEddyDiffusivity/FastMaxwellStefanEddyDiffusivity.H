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
      2. Provide interface for turbulent transport model

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
#include "compressibleMomentumTransportModel.H"
#include "fluidThermo.H"
#include "fluidReactionThermo.H"

//---------------------------------
// 4. FastMaxwellStefan headers
//---------------------------------
#include "FastMaxwellStefan.H"

//=============================================================================//
#ifndef FastMaxwellStefanEddyDiffusivity_H
#define FastMaxwellStefanEddyDiffusivity_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace turbulenceThermophysicalTransportModels
{

/*---------------------------------------------------------------------------*\
                Class FastMaxwellStefanEddyDiffusivity Declaration
\*---------------------------------------------------------------------------*/

template<class TurbulenceThermophysicalTransportModel>
class FastMaxwellStefanEddyDiffusivity
:
    public   TurbulenceThermophysicalTransportModel
{

    // Private data

            label nSpecies;

            FastMaxwellStefan MS;

protected:

    // Protected data

        // Model coefficients

            //- Turbulent Schmidt number []
            dimensionedScalar Sct_;

        // Model coefficients

            //- Turbulent Prandtl number []
            dimensionedScalar Prt_;

        // Fields

            //- Turbulent thermal diffusivity of enthalpy [kg/m/s]
            volScalarField alphat_;

    // Protected Member Functions

        virtual void correctAlphat();

public:

    typedef typename TurbulenceThermophysicalTransportModel::alphaField
        alphaField;

    typedef typename
        TurbulenceThermophysicalTransportModel::momentumTransportModel
        momentumTransportModel;

    typedef typename TurbulenceThermophysicalTransportModel::thermoModel
        thermoModel;


    //- Runtime type information
    TypeName("FastMaxwellStefanEddyDiffusivity");


    // Constructors

        //- Construct from a momentum transport model and a thermo model
        FastMaxwellStefanEddyDiffusivity
        (
            const momentumTransportModel& momentumTransport,
            const thermoModel& thermo
        );


    //- Destructor
    virtual ~FastMaxwellStefanEddyDiffusivity();

    // Member Functions

        //- Turbulent thermal diffusivity for enthalpy [kg/m/s]
        virtual tmp<volScalarField> alphat() const
        {
            return alphat_;
        }

        //- Turbulent thermal diffusivity for enthalpy for a patch [kg/m/s]
        virtual tmp<scalarField> alphat(const label patchi) const
        {
            return alphat()().boundaryField()[patchi];
        }

        //- Effective thermal turbulent diffusivity for temperature
        //  of mixture [W/m/K]
        virtual tmp<volScalarField> kappaEff() const
        {
            return this->thermo().kappaEff(alphat());
        }

        //- Effective thermal turbulent diffusivity for temperature
        //  of mixture for patch [W/m/K]
        virtual tmp<scalarField> kappaEff(const label patchi) const
        {
            return this->thermo().kappaEff
            (
                alphat(patchi),
                patchi
            );
        }

        //- Effective thermal turbulent diffusivity of mixture [kg/m/s]
        virtual tmp<volScalarField> alphaEff() const
        {
            return this->thermo().alphaEff(alphat());
        }

        //- Effective thermal turbulent diffusivity of mixture
        //  for patch [kg/m/s]
        virtual tmp<scalarField> alphaEff(const label patchi) const
        {
            return this->thermo().alphaEff
            (
                alphat(patchi),
                patchi
            );
        }

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

} // End namespace turbulenceThermophysicalTransportModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "FastMaxwellStefanEddyDiffusivity.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
