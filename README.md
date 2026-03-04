# Fast Multicomponent Transport Model 

  An improved multicomponent transport model for OpenFOAM-10 to accelerates the Maxwell-Stefan Computation.

## 1.  Introduction

  Maxwell-Stefan model provides more accurate description for molecular diffusion than
  Fickian model, and become important under certain situation for combustion, please see the reference below:
  
  1.  Dworkin, S.B.; Smooke, M.D.; Giovangigli, V. The Impact of Detailed Multicomponent Transport and 
      Thermal Diffusion Effects on Soot Formation in Ethylene/Air Flames. Proceedings of the Combustion 
      Institute 2009, 32, 1165–1172, doi:10.1016/j.proci.2008.05.061.

  2.  Fillo, A.J.; Schlup, J.; Blanquart, G. et al. Assessing the Impact of Multicomponent Diffusion in
      Direct Numerical Simulations of Premixed, High-Karlovitz, Turbulent Flames. Combustion and Flame 
      2021, 223, 216–229, doi:10.1016/j.combustflame.2020.09.013.
      
  3.  Chi, C.; Thévenin, D. Impact of Different Diffusion Models on NO Production in NH₃/H₂/Air Turbulent
      Flames Using DNS. In Proceedings of the Proceedings of the Global Power and Propulsion Society (GPPS)
      Xi’an21 Conference; Global Power and Propulsion Society, 2022; p. GPPS-TC-2021-3.

## 1.1 Key Advantages
  1. Mathematically consistent with OpenFOAM-10’s original formulation  
  2. No third-party dependencies  
  3. Significant performance improvement  

## 1.2 Limitations
  1. When using this model, the default species should be the last species in speciesTable, an example is 
  shown below:
  
  **physicalProperties**
  defaultSpecie N2;
  species       10 (H H2 O O2 OH H2O HO2 H2O2 AR N2);
  
  2. Do not support Soret effect currently(still under development).

## 1.3 Numerical consistency
  This code provides mathematical accuracy equivalent to OpenFOAM
  
## 2.  Installation
## 2.1 Prerequisites

  1. cmake >= 3.10.2  
  2. gcc >= 5.4.0  
  3. OpenFOAM-10  

## 2.2 Compilation
## 2.2.1   Option 1: using wmake
  If your glibc version is ≥ 2.22, you can build with the OpenFOAM wmake command,
  since glibc-2.22 provides vectorized mathematical functions. The installation steps are:
  1. Enter the `src` folder.
  2. In `Macro.H`, set `USE_LOCALFILE_  false`. 
  3. Source your OpenFOAM environment.  
  4. Run the `wmake`

## 2.2.2   Option 2: using Cmake and make

  1. Enter the `src` folder.
  2. Source your OpenFOAM environment.
  3. Create a `build` directory. 
        `mkdir build`
  4. Enter the `build` folder. 
        `cd build`
  5. Run CMake.
        `cmake ..`
  6. Build with Make.
        `make`  
  7. Copy the dynamic library `libFastFullMultiComponentTransport.so` to the folder `$FOAM_USER_LIBBIN`
  *  Note: When building on a compute cluster, CMake may pick the system’s GCC by default, 
     which can be too old (e.g., GCC 4.8.5 lacks C++14 support) and cause compilation failures.
     In that case, load a newer GCC and run:
    `cmake -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) ..`

## 2.2.3   Vectorized math function

  If your glibc version is **< 2.22**, the libmvec.so lacks and the vector math 
  function exp, pow or log cannot be used. In this situation, you can set 
  the macro `USE_LOCALFILE_ true` in Macro.H. The module will use 
  assembly code of vectorized math function of glibc 2.22 in asmCode folder.


## 3.  Usage
## 3.1 controlDict
  add this library in Case/system/controlDict
  **controlDict**
    libs
    (
        "libFastFullMultiComponentTransport.so"
    );

## 3.2 thermophysicalTransport
  add this text in Case/constant/thermophysicalTransport for laminar flow
  **thermophysicalTransport**
    laminar
    {
	model           FastMaxwellStefanFourier;

	D // [m^2/s] 
	{
            #include "2DInterpolationTableText"
	}
    }
    
  add this text in Case/constant/thermophysicalTransport for turbulent LES or RAS flow
  **thermophysicalTransport**
    //LES
    RAS
    {
        model           FastMaxwellStefanEddyDiffusivity;

        Prt             0.7;
        Sct             0.7;

        D // [m^2/s] 
        {
            #include "2DInterpolationTableText"
        }
    }

## 3.2.1   2DInterpolation

  This model support 2D interpolation function for computing binary diffusion coefficient
  using OpenFOAM function2.
  
  "2DInterpolationTableText" contains a 2D interpolation table for evaluating binary 
  diffusion coefficients of species pairs using temperature and pressure as lookup keys.
  Example entries:
  "H-H" denotes the binary diffusion coefficient between H₂ and H species.
  "80000.0" and "120000.0" define the minimum and maximum pressure limits (Pa).
  "285.0" and "3000.0" define the minimum and maximum temperature limits (K).
  Ensure that lookup values remain within these physical ranges; otherwise OpenFOAM 
  will report a runtime error and terminate the simulation.
  **2DInterpolationTableText**
    H2-H
    {
        type   uniformTable;
        low   (80000.0   285.0);
        high  (120000.0   3000.0);

        values
        2	2
        (
            (0.0003506978264889336	0.00040962147615834413)
            (0.0002337985509926224	0.013085997974807187)
        );
    }
  
  If you decide to use logPolynomial,This file can be blank.
  
  Try the code in this repository, it can automatically generate a two-dimensional 
  interpolation table using Cantera:
  
    https://github.com/yuchenzh/DDOF
    
## 3.2.2   logPolynomial

  This model support log polynomial function for computing binary diffusion coefficient
  using CHEMKIN format:
  
  1. Open ANSYS CHEMKIN premixed freely propagation laminar flame models
  2. Click 'pre-processing','Edit Chemistry Set'
  3. Find entry 'Process Transport Properties', select Fit with Verbose Output
  4. Click 'save' to lock the edit option, click 'Run Pre-Processor'
  5. Find the '*tran.out' text, rename it as 'CKtransport' and put it into Case/constant/

  If the model detect the file named 'CKtransport', it will automatically use logPolynomial
  to compute binary diffusion coefficient.

## License

    This OpenFOAM library is under the GNU General Public License.  

## Contact

    Maintainer: Zixin Chi  
    Email: chizixin@buaa.edu.cn  
    Issues: Please use the [GitHub Issues] page for bug reports and questions.  



