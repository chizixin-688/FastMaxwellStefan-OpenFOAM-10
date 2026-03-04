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
#include <iostream>
#include <chrono>
#include <stdlib.h>

//---------------------------------
// 2. OpenFOAM-10 headers
//---------------------------------
#include "Function2Evaluate.H"

//---------------------------------
// 3. FastMaxwellStefan headers
//---------------------------------
#include "FastMaxwellStefanFourier.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarThermophysicalTransportModels
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicThermophysicalTransportModel>
FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::FastMaxwellStefanFourier
(
    const momentumTransportModel& momentumTransport,
    const thermoModel& thermo
)
:
    laminarThermophysicalTransportModel<BasicThermophysicalTransportModel>
    (
        typeName,
        momentumTransport,
        thermo
    ),
    nSpecies(this->thermo().composition().species().size()),
    MS
    (
        this->coeffDict_,
        this->thermo().T().mesh(),
        this->thermo().composition(),
        this->thermo().T(),
        this->thermo().p(),
        this->momentumTransport().rho()
    )
{
    const basicSpecieMixture& composition = this->thermo().composition();

    Info << "============================================================================================================"<<endl;
    Info << "To use the FastMaxwellStefanFourier, the default specie must be the last specie in speciesTable, checking..."<<endl;
    Info << "============================================================================================================"<<endl;

    const label d = composition.defaultSpecie();

    if(d != composition.Y().size()-1)
    {
        FatalErrorInFunction
            << "The default specie is not the last specie in speciesTable" << Foam::abort(FatalError);
    }

    this->read();

    this->correct();
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
template<class BasicThermophysicalTransportModel>
FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::~FastMaxwellStefanFourier()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicThermophysicalTransportModel>
bool
FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::read()
{
    return true;
}

template<class BasicThermophysicalTransportModel>
tmp<volScalarField> FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::DEff
(
    const volScalarField& Yi
) const
{
    const basicSpecieMixture& composition = this->thermo().composition();

    return volScalarField::New
    (
        "DEff",
        this->momentumTransport().rho()*MS.Dii()[composition.index(Yi)]
    );
}

template<class BasicThermophysicalTransportModel>
tmp<scalarField> FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::DEff
(
    const volScalarField& Yi,
    const label patchi
) const
{
    const basicSpecieMixture& composition = this->thermo().composition();

    return
        this->momentumTransport().rho().boundaryField()[patchi]
       *MS.Dii()[composition.index(Yi)].boundaryField()[patchi];
}

template<class BasicThermophysicalTransportModel>
tmp<surfaceScalarField>
FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::q() const
{
    tmp<surfaceScalarField> tmpq
    (
        surfaceScalarField::New
        (
            IOobject::groupName
            (
                "q",
                this->momentumTransport().alphaRhoPhi().group()
            ),
           -fvc::interpolate(this->alpha()*this->kappaEff())
           *fvc::snGrad(this->thermo().T())
        )
    );

    const basicSpecieMixture& composition = this->thermo().composition();

    const PtrList<volScalarField>& Yfields = composition.Y();

    if (Yfields.size())
    {
        surfaceScalarField sumJ
        (
            surfaceScalarField::New
            (
                "sumJ",
                Yfields[0].mesh(),
                dimensionedScalar(dimMass/dimArea/dimTime, 0)
            )
        );

        surfaceScalarField sumJh
        (
            surfaceScalarField::New
            (
                "sumJh",
                Yfields[0].mesh(),
                dimensionedScalar(sumJ.dimensions()*dimEnergy/dimMass, 0)
            )
        );


        for(label i=0; i<this->nSpecies-1; i++)
        {
            const volScalarField hi
            (
                composition.Hs(i, this->thermo().p(), this->thermo().T())
            );

            const surfaceScalarField ji(this->j(Yfields[i]));
            sumJ += ji;

            sumJh += ji*fvc::interpolate(hi);
        }

        {
            const label i = this->nSpecies-1;

            const volScalarField hi
            (
                composition.Hs(i, this->thermo().p(), this->thermo().T())
            );

            sumJh -= sumJ*fvc::interpolate(hi);
        }

        tmpq.ref() += sumJh;
    }

    return tmpq;
}

template<class BasicThermophysicalTransportModel>
tmp<fvScalarMatrix> FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::divq
(
    volScalarField& he
) const
{
    tmp<fvScalarMatrix> tmpDivq
    (
        fvm::Su
        (
            -fvc::laplacian(this->alpha()*this->kappaEff(), this->thermo().T()),
            he
        )
    );

    const basicSpecieMixture& composition = this->thermo().composition();
    const label d = composition.defaultSpecie();

    const PtrList<volScalarField>& Yfield = composition.Y();

    tmpDivq.ref() -=
        correction(fvm::laplacian(this->alpha()*this->alphaEff(), he));

    surfaceScalarField sumJ
    (
        surfaceScalarField::New
        (
            "sumJ",
            he.mesh(),
            dimensionedScalar(dimMass/dimArea/dimTime, 0)
        )
    );

    surfaceScalarField sumJh
    (
        surfaceScalarField::New
        (
            "sumJh",
            he.mesh(),
            dimensionedScalar(sumJ.dimensions()*he.dimensions(), 0)
        )
    );

    for(label i=0; i<this->nSpecies-1; i++)
    {
        const volScalarField hi
        (
            composition.Hs(i, this->thermo().p(), this->thermo().T())
        );

        const surfaceScalarField ji(this->j(Yfield[i]));
        sumJ += ji;

        sumJh += ji*fvc::interpolate(hi);
    }

    {
        const label i = d;

        const volScalarField hi
        (
            composition.Hs(i, this->thermo().p(), this->thermo().T())
        );

        sumJh -= sumJ*fvc::interpolate(hi);
    }

    tmpDivq.ref() += fvc::div(sumJh*he.mesh().magSf());

    return tmpDivq;
}


template<class BasicThermophysicalTransportModel>
tmp<surfaceScalarField> FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::j
(
    const volScalarField& Yi
) const
{
    const basicSpecieMixture& composition = this->thermo().composition();
    const label d = composition.defaultSpecie();

    if (composition.index(Yi) == d)
    {
        const PtrList<volScalarField>& Yfield = composition.Y();

        tmp<surfaceScalarField> tjd
        (
            surfaceScalarField::New
            (
                IOobject::groupName
                (
                    "j" + name(d),
                    this->momentumTransport().alphaRhoPhi().group()
                ),
                Yi.mesh(),
                dimensionedScalar(dimMass/dimArea/dimTime, 0)
            )
        );

        surfaceScalarField& jd = tjd.ref();

        for(label i=0; i<this->nSpecies-1; i++)
        {
            jd -= this->j(Yfield[i]);
        }

        return tjd;
    }
    else
    {
        return

        surfaceScalarField::New
        (
            IOobject::groupName
            (
                "j(" + Yi.name() + ')',
                this->momentumTransport().alphaRhoPhi().group()
            ),
           -fvc::interpolate(this->alpha()*this->DEff(Yi))
           *fvc::snGrad(Yi)
        )
        + MS.jexp()[composition.index(Yi)];
    }
}

template<class BasicThermophysicalTransportModel>
tmp<fvScalarMatrix> FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::divj
(
    volScalarField& Yi
) const
{
    const basicSpecieMixture& composition = this->thermo().composition();
    return
        -fvm::laplacian(this->alpha()*this->DEff(Yi), Yi)
      + fvc::div(MS.jexp()[composition.index(Yi)]*Yi.mesh().magSf());
}

template<class BasicThermophysicalTransportModel>
void FastMaxwellStefanFourier<BasicThermophysicalTransportModel>::correct()
{
    if(MS.logPolynomial()==true)
    {
        MS.correctUsingPolynomial();
    }
    else
    {
        MS.correctUsingInterpolation();
    }
    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace BasicThermophysicalTransportModels
} // End namespace Foam

// ************************************************************************* //
