/*---------------------------------------------------------------------------*\
  This interface file is derived from the OpenFOAM solver framework.

  Original OpenFOAM implementation:
      OpenFOAM Foundation

  Modified by:
      Zixin Chi, 2026

  Purpose of modification:
  Registers custom high-performance thermophysical transport models into the 
  OpenFOAM runtime selection framework for laminar, RAS, and LES simulations.

  All modifications comply with GNU GPL licensing requirements.

\*---------------------------------------------------------------------------*/


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#include "fluidReactionThermophysicalTransportModels.H"

// -------------------------------------------------------------------------- //
// Laminar models
// -------------------------------------------------------------------------- //
#include "FastMaxwellStefanFourier.H"
makeLaminarThermophysicalTransportModel(FastMaxwellStefanFourier)

// -------------------------------------------------------------------------- //
// RAS models
// -------------------------------------------------------------------------- //

#include "FastMaxwellStefanEddyDiffusivity.H"
makeRASLESThermophysicalTransportModel(RAS, FastMaxwellStefanEddyDiffusivity);
// -------------------------------------------------------------------------- //
// LES models
// -------------------------------------------------------------------------- //

#include "FastMaxwellStefanEddyDiffusivity.H"
makeRASLESThermophysicalTransportModel(LES, FastMaxwellStefanEddyDiffusivity);

// ************************************************************************* //
