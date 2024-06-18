/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef AFFT_BACKEND_H
#define AFFT_BACKEND_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Backend type
typedef uint16_t afft_Backend;

/// @brief Backend enumeration
enum
{
  afft_Backend_clfft     = (1 << 0), ///< clFFT
  afft_Backend_cufft     = (1 << 1), ///< cuFFT
  afft_Backend_fftw3     = (1 << 2), ///< FFTW3
  afft_Backend_heffte    = (1 << 3), ///< HeFFTe
  afft_Backend_hipfft    = (1 << 4), ///< hipFFT
  afft_Backend_mkl       = (1 << 5), ///< Intel MKL
  afft_Backend_pocketfft = (1 << 6), ///< PocketFFT
  afft_Backend_rocfft    = (1 << 7), ///< rocFFT
  afft_Backend_vkfft     = (1 << 8), ///< VkFFT
};

/// @brief Backend count
#define AFFT_BACKEND_COUNT 9

/// @brief Backend mask type
typedef uint16_t afft_BackendMask;

/// @brief Backend mask enumeration
enum
{
  afft_BackendMask_empty = 0,          ///< Empty mask
  afft_BackendMask_all   = UINT16_MAX, ///< All backends
};

/// @brief Select strategy type
typedef uint8_t afft_SelectStrategy;

/// @brief Select strategy enumeration
enum
{
  afft_SelectStrategy_first, ///< Select the first available backend
  afft_SelectStrategy_best,  ///< Select the best available backend
};

/**********************************************************************************************************************/
// clFFT
/**********************************************************************************************************************/
/// @brief clFFT backend parameters for spst gpu architecture
typedef struct
{
  bool useFastMath;
} afft_spst_gpu_clfft_Parameters;

/**********************************************************************************************************************/
// cuFFT
/**********************************************************************************************************************/
/// @brief cuFFT workspace policy type
typedef uint8_t afft_cufft_WorkspacePolicy;

/// @brief cuFFT workspace policy enumeration
enum
{
  afft_cufft_WorkspacePolicy_performance, ///< Use the workspace for performance
  afft_cufft_WorkspacePolicy_minimal,     ///< Use the minimal workspace
  afft_cufft_WorkspacePolicy_user,        ///< Use the user-defined workspace size
};

/// @brief cuFFT backend parameters for spst gpu architecture
typedef struct
{
  afft_cufft_WorkspacePolicy workspacePolicy;   ///< Workspace policy
  bool                       usePatientJit;     ///< Use patient JIT
  size_t                     userWorkspaceSize; ///< User-defined workspace size
} afft_spst_gpu_cufft_Parameters;

/// @brief cuFFT backend parameters for spmt gpu architecture
typedef struct
{
  bool usePatientJit; ///< Use patient JIT
} afft_spmt_gpu_cufft_Parameters;

/// @brief cuFFT backend parameters for mpst gpu architecture
typedef struct
{
  bool usePatientJit; ///< Use patient JIT
} afft_mpst_gpu_cufft_Parameters;

/**********************************************************************************************************************/
// FFTW3
/**********************************************************************************************************************/
/// @brief FFTW3 planner flag type
typedef uint8_t afft_fftw3_PlannerFlag;

/// @brief FFTW3 planner flag enumeration
enum
{
  afft_fftw3_PlannerFlag_estimate,        ///< Estimate plan flag
  afft_fftw3_PlannerFlag_measure,         ///< Measure plan flag
  afft_fftw3_PlannerFlag_patient,         ///< Patient plan flag
  afft_fftw3_PlannerFlag_exhaustive,      ///< Exhaustive planner flag
  afft_fftw3_PlannerFlag_estimatePatient, ///< Estimate and patient plan flag
};

/// @brief FFTW3 backend parameters for spst cpu architecture
typedef struct
{
  afft_fftw3_PlannerFlag plannerFlag;       ///< FFTW3 planner flag
  bool                   conserveMemory;    ///< Conserve memory flag
  bool                   wisdomOnly;        ///< Wisdom only flag
  bool                   allowLargeGeneric; ///< Allow large generic flag
  bool                   allowPruning;      ///< Allow pruning flag
  double                 timeLimit;         ///< Time limit for the planner
} afft_spst_cpu_fftw3_Parameters;

/// @brief FFTW3 backend parameters for mpst cpu architecture
typedef struct
{
  afft_fftw3_PlannerFlag plannerFlag;       ///< FFTW3 planner flag
  bool                   conserveMemory;    ///< Conserve memory flag
  bool                   wisdomOnly;        ///< Wisdom only flag
  bool                   allowLargeGeneric; ///< Allow large generic flag
  bool                   allowPruning;      ///< Allow pruning flag
  double                 timeLimit;         ///< Time limit for the planner
  size_t                 blockSize;         ///< Decomposition block size
} afft_mpst_cpu_fftw3_Parameters;

/**********************************************************************************************************************/
// HeFFTe
/**********************************************************************************************************************/
/// @brief HeFFTe cpu backend type
typedef uint8_t afft_heffte_cpu_Backend;

/// @brief HeFFTe cpu backend enumeration
enum
{
  afft_heffte_cpu_Backend_fftw3, ///< FFTW3 backend
  afft_heffte_cpu_Backend_mkl,   ///< MKL backend
};

/// @brief HeFFTe gpu backend type
typedef uint8_t afft_heffte_gpu_Backend;

/// @brief HeFFTe gpu backend enumeration
enum
{
  afft_heffte_gpu_Backend_cufft,  ///< cuFFT backend
  afft_heffte_gpu_Backend_rocfft, ///< rocFFT backend
};

/// @brief HeFFTe backend parameters for mpst cpu architecture
typedef struct
{
  afft_heffte_cpu_Backend backend;     ///< HeFFTe backend
  bool                    useReorder;  ///< Use reorder flag
  bool                    useAllToAll; ///< Use all-to-all flag
  bool                    usePencils;  ///< Use pencils flag
} afft_mpst_cpu_heffte_Parameters;

/// @brief HeFFTe backend parameters for mpst gpu architecture
typedef struct
{
  afft_heffte_gpu_Backend backend;     ///< HeFFTe backend
  bool                    useReorder;  ///< Use reorder flag
  bool                    useAllToAll; ///< Use all-to-all flag
  bool                    usePencils;  ///< Use pencils flag
} afft_mpst_gpu_heffte_Parameters;

/**********************************************************************************************************************/
// Backend parameters for spst distribution
/**********************************************************************************************************************/
/// @brief Backend parameters for spst cpu architecture
typedef struct
{
  afft_SelectStrategy            strategy;  ///< Select strategy
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_spst_cpu_fftw3_Parameters fftw3;     ///< FFTW3 parameters
} afft_spst_cpu_BackendParameters;

/// @brief Backend parameters for spst gpu architecture
typedef struct
{
  afft_SelectStrategy            strategy;  ///< Select strategy
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_spst_gpu_clfft_Parameters clfft;     ///< clFFT parameters
  afft_spst_gpu_cufft_Parameters cufft;     ///< cuFFT parameters
} afft_spst_gpu_BackendParameters;

/**********************************************************************************************************************/
// Backend parameters for spmt distribution
/**********************************************************************************************************************/
/// @brief Backend parameters for spmt gpu architecture
typedef struct
{
  afft_SelectStrategy            strategy;  ///< Select strategy
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_spmt_gpu_cufft_Parameters cufft;     ///< cuFFT parameters
} afft_spmt_gpu_BackendParameters;

/**********************************************************************************************************************/
// Backend parameters for mpst distribution
/**********************************************************************************************************************/
/// @brief Backend parameters for mpst cpu architecture
typedef struct
{
  afft_SelectStrategy             strategy;  ///< Select strategy
  afft_BackendMask                mask;      ///< Backend mask
  size_t                          orderSize; ///< Number of backends in the order
  const afft_Backend*             order;     ///< Order of the backends
  afft_mpst_cpu_fftw3_Parameters  fftw3;     ///< FFTW3 parameters
  afft_mpst_cpu_heffte_Parameters heffte;    ///< HeFFTe parameters
} afft_mpst_cpu_BackendParameters;

/// @brief Backend parameters for mpst gpu architecture
typedef struct
{
  afft_SelectStrategy             strategy;  ///< Select strategy
  afft_BackendMask                mask;      ///< Backend mask
  size_t                          orderSize; ///< Number of backends in the order
  const afft_Backend*             order;     ///< Order of the backends
  afft_mpst_gpu_cufft_Parameters  cufft;     ///< cuFFT parameters
  afft_mpst_gpu_heffte_Parameters heffte;    ///< HeFFTe parameters
} afft_mpst_gpu_BackendParameters;

/**********************************************************************************************************************/
// General backend parameters
/**********************************************************************************************************************/

/// @brief Backend parameters structure
typedef struct
{
  union
  {
    afft_spst_cpu_BackendParameters spstCpu;
    afft_spst_gpu_BackendParameters spstGpu;
    afft_spmt_gpu_BackendParameters spmtGpu;
    afft_mpst_cpu_BackendParameters mpstCpu;
    afft_mpst_gpu_BackendParameters mpstGpu;
  };
  afft_Target                       target;
  afft_Distribution                 distribution;
} afft_BackendParameters;

typedef afft_spst_gpu_clfft_Parameters afft_gpu_clfft_Parameters;
typedef afft_spst_gpu_cufft_Parameters afft_gpu_cufft_Parameters;
typedef afft_spst_cpu_fftw3_Parameters afft_cpu_fftw3_Parameters;

typedef afft_spst_cpu_BackendParameters afft_cpu_BackendParameters;
typedef afft_spst_gpu_BackendParameters afft_gpu_BackendParameters;

/// @brief Feedback structure
typedef struct
{
  afft_Backend backend;      ///< Backend
  const char*  message;      ///< Message from the backend
  double       measuredTime; ///< Measured time in seconds
} afft_Feedback;

/**********************************************************************************************************************/
// Private functions
/**********************************************************************************************************************/
static inline afft_BackendParameters _afft_makeBackendParametersSpstCpu(afft_spst_cpu_BackendParameters params)
{
  afft_BackendParameters result;
  result.spstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_BackendParameters _afft_makeBackendParametersSpstGpu(afft_spst_gpu_BackendParameters params)
{
  afft_BackendParameters result;
  result.spstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_BackendParameters _afft_makeBackendParametersSpmtGpu(afft_spmt_gpu_BackendParameters params)
{
  afft_BackendParameters result;
  result.spmtGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spmt;

  return result;
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstCpu(afft_mpst_cpu_BackendParameters params)
{
  afft_BackendParameters result;
  result.mpstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstGpu(afft_mpst_gpu_BackendParameters params)
{
  afft_BackendParameters result;
  result.mpstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_BackendParameters _afft_makeBackendParametersAny(afft_BackendParameters params)
{
  return params;
}

/**********************************************************************************************************************/
// Public functions
/**********************************************************************************************************************/
#ifdef __cplusplus
} // extern "C"

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_spst_cpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpstCpu(params);
}

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_spst_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpstGpu(params);
}

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_spmt_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpmtGpu(params);
}

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_mpst_cpu_BackendParameters params)
{
  return _afft_makeBackendParametersMpstCpu(params);
}

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_mpst_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersMpstGpu(params);
}

/**
 * @brief Make backend parameters
 * @param params Backend parameters
 * @return Backend parameters
 */
static inline afft_BackendParameters afft_makeBackendParameters(afft_BackendParameters params)
{
  return _afft_makeBackendParametersAny(params);
}

extern "C"
{
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  /**
   * @brief Make backend parameters
   * @param params Backend parameters
   * @return Backend parameters
   */
# define afft_makeBackendParameters(params) _Generic((params), \
    afft_spst_cpu_BackendParameters: _afft_makeBackendParametersSpstCpu, \
    afft_spst_gpu_BackendParameters: _afft_makeBackendParametersSpstGpu, \
    afft_spmt_gpu_BackendParameters: _afft_makeBackendParametersSpmtGpu, \
    afft_mpst_cpu_BackendParameters: _afft_makeBackendParametersMpstCpu, \
    afft_mpst_gpu_BackendParameters: _afft_makeBackendParametersMpstGpu, \
    afft_BackendParameters:          _afft_makeBackendParametersAny)(params)
#endif

/**
 * @brief Get the name of the backend
 * @param backend Backend
 * @return Name of the backend
 */
const char* getBackendName(afft_Backend backend);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_BACKEND_H */
