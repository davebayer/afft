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
/// @brief clFFT backend parameters for opencl target
typedef struct afft_clfft_opencl_Parameters afft_clfft_opencl_Parameters;

/// @brief clFFT backend parameters for opencl target
struct afft_clfft_opencl_Parameters
{
  bool useFastMath;
};

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

/// @brief cuFFT backend parameters for cuda target
typedef struct afft_cufft_cuda_Parameters afft_cufft_cuda_Parameters;

/// @brief cuFFT backend parameters for mpi cuda target
typedef struct afft_cufft_mpi_cuda_Parameters afft_cufft_mpi_cuda_Parameters;

/// @brief cuFFT backend parameters for cuda target
struct afft_cufft_cuda_Parameters
{
  afft_cufft_WorkspacePolicy workspacePolicy;   ///< Workspace policy
  bool                       usePatientJit;     ///< Use patient JIT
  size_t                     userWorkspaceSize; ///< User-defined workspace size
};

/// @brief cuFFT backend parameters for mpi cuda target
struct afft_cufft_mpi_cuda_Parameters
{
  bool usePatientJit; ///< Use patient JIT
};

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

/// @brief FFTW3 backend parameters for cpu target
typedef struct afft_fftw3_cpu_Parameters afft_fftw3_cpu_Parameters;

/// @brief FFTW3 backend parameters for mpi cpu target
typedef struct afft_fftw3_mpi_cpu_Parameters afft_fftw3_mpi_cpu_Parameters;

/// @brief No time limit for the planner
#define AFFT_FFTW3_NO_TIME_LIMIT -1.0

/// @brief FFTW3 backend parameters for cpu target
struct afft_fftw3_cpu_Parameters
{
  afft_fftw3_PlannerFlag plannerFlag;       ///< FFTW3 planner flag
  bool                   conserveMemory;    ///< Conserve memory flag
  bool                   wisdomOnly;        ///< Wisdom only flag
  bool                   allowLargeGeneric; ///< Allow large generic flag
  bool                   allowPruning;      ///< Allow pruning flag
  double                 timeLimit;         ///< Time limit for the planner
};

/// @brief FFTW3 backend parameters for mpi cpu target
struct afft_fftw3_mpi_cpu_Parameters
{
  afft_fftw3_PlannerFlag plannerFlag;       ///< FFTW3 planner flag
  bool                   conserveMemory;    ///< Conserve memory flag
  bool                   wisdomOnly;        ///< Wisdom only flag
  bool                   allowLargeGeneric; ///< Allow large generic flag
  bool                   allowPruning;      ///< Allow pruning flag
  double                 timeLimit;         ///< Time limit for the planner
  afft_Size              blockSize;         ///< Decomposition block size
};

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

/// @brief HeFFTe cuda backend type
typedef uint8_t afft_heffte_cuda_Backend;

/// @brief HeFFTe gpu backend enumeration
enum
{
  afft_heffte_gpu_Backend_cufft,  ///< cuFFT backend
};

/// @brief HeFFTe hip backend type
typedef uint8_t afft_heffte_hip_Backend;

/// @brief HeFFTe hip backend enumeration
enum
{
  afft_heffte_hip_Backend_rocfft, ///< rocFFT backend
};

/// @brief HeFFTe backend parameters for mpi cpu target
typedef struct afft_heffte_mpi_cpu_Parameters afft_heffte_mpi_cpu_Parameters;

/// @brief HeFFTe backend parameters for mpi cuda target
typedef struct afft_heffte_mpi_cuda_Parameters afft_heffte_mpi_cuda_Parameters;

/// @brief HeFFTe backend parameters for mpi hip target
typedef struct afft_heffte_mpi_hip_Parameters afft_heffte_mpi_hip_Parameters;

/// @brief HeFFTe backend parameters for mpi cpu target
struct afft_heffte_mpi_cpu_Parameters
{
  afft_heffte_cpu_Backend backend;     ///< HeFFTe backend
  bool                    useReorder;  ///< Use reorder flag
  bool                    useAllToAll; ///< Use all-to-all flag
  bool                    usePencils;  ///< Use pencils flag
};

/// @brief HeFFTe backend parameters for mpi cuda target
struct afft_heffte_mpi_cuda_Parameters
{
  afft_heffte_cuda_Backend backend;     ///< HeFFTe backend
  bool                     useReorder;  ///< Use reorder flag
  bool                     useAllToAll; ///< Use all-to-all flag
  bool                     usePencils;  ///< Use pencils flag
};

/// @brief HeFFTe backend parameters for mpi hip target
struct afft_heffte_mpi_hip_Parameters
{
  afft_heffte_hip_Backend backend;     ///< HeFFTe backend
  bool                    useReorder;  ///< Use reorder flag
  bool                    useAllToAll; ///< Use all-to-all flag
  bool                    usePencils;  ///< Use pencils flag
};

/**********************************************************************************************************************/
// Backend parameters for single process targets
/**********************************************************************************************************************/
/// @brief Backend parameters for cpu target
typedef struct afft_cpu_BackendParameters afft_cpu_BackendParameters;

/// @brief Backend parameters for cuda target
typedef struct afft_cuda_BackendParameters afft_cuda_BackendParameters;

/// @brief Backend parameters for hip target
typedef struct afft_hip_BackendParameters afft_hip_BackendParameters;

/// @brief Backend parameters for opencl target
typedef struct afft_opencl_BackendParameters afft_opencl_BackendParameters;

/// @brief Backend parameters for cpu target
struct afft_cpu_BackendParameters
{
  afft_SelectStrategy       strategy;  ///< Select strategy
  afft_BackendMask          mask;      ///< Backend mask
  size_t                    orderSize; ///< Number of backends in the order
  const afft_Backend*       order;     ///< Order of the backends
  afft_fftw3_cpu_Parameters fftw3;     ///< FFTW3 parameters
};

/// @brief Backend parameters for cuda target
struct afft_cuda_BackendParameters
{
  afft_SelectStrategy        strategy;  ///< Select strategy
  afft_BackendMask           mask;      ///< Backend mask
  size_t                     orderSize; ///< Number of backends in the order
  const afft_Backend*        order;     ///< Order of the backends
  afft_cufft_cuda_Parameters cufft;     ///< cuFFT parameters
};

/// @brief Backend parameters for hip target
struct afft_hip_BackendParameters
{
  afft_SelectStrategy        strategy;  ///< Select strategy
  afft_BackendMask           mask;      ///< Backend mask
  size_t                     orderSize; ///< Number of backends in the order
  const afft_Backend*        order;     ///< Order of the backends
};

/// @brief Backend parameters for cuda target
struct afft_opencl_BackendParameters
{
  afft_SelectStrategy          strategy;  ///< Select strategy
  afft_BackendMask             mask;      ///< Backend mask
  size_t                       orderSize; ///< Number of backends in the order
  const afft_Backend*          order;     ///< Order of the backends
  afft_clfft_opencl_Parameters clfft;     ///< clFFT parameters
};

/**********************************************************************************************************************/
// Backend parameters for mpi targets
/**********************************************************************************************************************/
/// @brief Backend parameters for mpi cpu target
typedef struct afft_mpi_cpu_BackendParameters afft_mpi_cpu_BackendParameters;

/// @brief Backend parameters for mpi cuda target
typedef struct afft_mpi_cuda_BackendParameters afft_mpi_cuda_BackendParameters;

/// @brief Backend parameters for mpi hip target
typedef struct afft_mpi_hip_BackendParameters afft_mpi_hip_BackendParameters;

/// @brief Backend parameters for mpi opencl target
typedef struct afft_mpi_opencl_BackendParameters afft_mpi_opencl_BackendParameters;

/// @brief Backend parameters for cpu target
struct afft_mpi_cpu_BackendParameters
{
  afft_SelectStrategy            strategy;  ///< Select strategy
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_fftw3_mpi_cpu_Parameters  fftw3;     ///< FFTW3 parameters
  afft_heffte_mpi_cpu_Parameters heffte;    ///< HeFFTe parameters
};

/// @brief Backend parameters for cuda target
struct afft_mpi_cuda_BackendParameters
{
  afft_SelectStrategy             strategy;  ///< Select strategy
  afft_BackendMask                mask;      ///< Backend mask
  size_t                          orderSize; ///< Number of backends in the order
  const afft_Backend*             order;     ///< Order of the backends
  afft_cufft_mpi_cuda_Parameters  cufft;     ///< cuFFT parameters
  afft_heffte_mpi_cuda_Parameters heffte;    ///< HeFFTe parameters
};

/// @brief Backend parameters for hip target
struct afft_mpi_hip_BackendParameters
{
  afft_SelectStrategy            strategy;  ///< Select strategy
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_heffte_mpi_hip_Parameters heffte;    ///< HeFFTe parameters
};

/// @brief Backend parameters for cuda target
struct afft_mpi_opencl_BackendParameters
{
  afft_SelectStrategy          strategy;  ///< Select strategy
  afft_BackendMask             mask;      ///< Backend mask
  size_t                       orderSize; ///< Number of backends in the order
  const afft_Backend*          order;     ///< Order of the backends
};

/// @brief Feedback structure
typedef struct
{
  afft_Backend backend;      ///< Backend
  const char*  message;      ///< Message from the backend
  double       measuredTime; ///< Measured time in seconds
} afft_Feedback;

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
