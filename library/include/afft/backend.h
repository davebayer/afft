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
typedef uint8_t afft_Backend;

/// @brief Backend enumeration
#define afft_Backend_clfft     (afft_Backend)0 ///< clFFT
#define afft_Backend_cufft     (afft_Backend)1 ///< cuFFT
#define afft_Backend_fftw3     (afft_Backend)2 ///< FFTW3
#define afft_Backend_heffte    (afft_Backend)3 ///< HeFFTe
#define afft_Backend_hipfft    (afft_Backend)4 ///< hipFFT
#define afft_Backend_mkl       (afft_Backend)5 ///< Intel MKL
#define afft_Backend_pocketfft (afft_Backend)6 ///< PocketFFT
#define afft_Backend_rocfft    (afft_Backend)7 ///< rocFFT
#define afft_Backend_vkfft     (afft_Backend)8 ///< VkFFT

/// @brief Backend count
#define AFFT_BACKEND_COUNT 9

/// @brief Backend mask type
typedef uint16_t afft_BackendMask;

/// @brief Backend mask enumeration
#define afft_BackendMask_empty     (afft_BackendMask)0                             ///< Empty mask
#define afft_BackendMask_clfft     (afft_BackendMask)(1 << afft_Backend_clfft)     ///< clFFT mask
#define afft_BackendMask_cufft     (afft_BackendMask)(1 << afft_Backend_cufft)     ///< cuFFT mask
#define afft_BackendMask_fftw3     (afft_BackendMask)(1 << afft_Backend_fftw3)     ///< FFTW3 mask
#define afft_BackendMask_heffte    (afft_BackendMask)(1 << afft_Backend_heffte)    ///< HeFFTe mask
#define afft_BackendMask_hipfft    (afft_BackendMask)(1 << afft_Backend_hipfft)    ///< hipFFT mask
#define afft_BackendMask_mkl       (afft_BackendMask)(1 << afft_Backend_mkl)       ///< Intel MKL mask
#define afft_BackendMask_pocketfft (afft_BackendMask)(1 << afft_Backend_pocketfft) ///< PocketFFT mask
#define afft_BackendMask_rocfft    (afft_BackendMask)(1 << afft_Backend_rocfft)    ///< rocFFT mask
#define afft_BackendMask_vkfft     (afft_BackendMask)(1 << afft_Backend_vkfft)     ///< VkFFT mask
#define afft_BackendMask_all       (afft_BackendMask)UINT16_MAX                    ///< All backends

/**
 * @brief Make backend mask from backend
 * @param[in] backend Backend
 * @return Backend mask
 */
static inline afft_BackendMask afft_makeBackendMask(afft_Backend backend)
{
  return (afft_BackendMask)(1 << backend);
}

/// @brief Select strategy type
typedef uint8_t afft_SelectStrategy;

/// @brief Select strategy enumeration
#define afft_SelectStrategy_first (afft_SelectStrategy)0 ///< Select the first available backend
#define afft_SelectStrategy_best  (afft_SelectStrategy)1 ///< Select the best available backend

/// @brief Workspace type
typedef uint8_t afft_Workspace;

/// @brief Workspace enumeration
#define afft_Workspace_any            (afft_Workspace)0 ///< Any workspace
#define afft_Workspace_none           (afft_Workspace)1 ///< No workspace
#define afft_Workspace_internal       (afft_Workspace)2 ///< internal workspace
#define afft_Workspace_external       (afft_Workspace)3 ///< external workspace
#define afft_Workspace_enlargedBuffer (afft_Workspace)4 ///< enlarged buffer

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
#define afft_cufft_WorkspacePolicy_performance (afft_cufft_WorkspacePolicy)0 ///< Use the workspace for performance
#define afft_cufft_WorkspacePolicy_minimal     (afft_cufft_WorkspacePolicy)1 ///< Use the minimal workspace
#define afft_cufft_WorkspacePolicy_user        (afft_cufft_WorkspacePolicy)2 ///< Use the user-defined workspace size

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
#define afft_fftw3_PlannerFlag_estimate        (afft_fftw3_PlannerFlag)0 ///< Estimate plan flag
#define afft_fftw3_PlannerFlag_measure         (afft_fftw3_PlannerFlag)1 ///< Measure plan flag
#define afft_fftw3_PlannerFlag_patient         (afft_fftw3_PlannerFlag)2 ///< Patient plan flag
#define afft_fftw3_PlannerFlag_exhaustive      (afft_fftw3_PlannerFlag)3 ///< Exhaustive planner flag
#define afft_fftw3_PlannerFlag_estimatePatient (afft_fftw3_PlannerFlag)4 ///< Estimate and patient plan flag

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
#define afft_heffte_cpu_Backend_fftw3 (afft_heffte_cpu_Backend)0 ///< FFTW3 backend
#define afft_heffte_cpu_Backend_mkl   (afft_heffte_cpu_Backend)1 ///< MKL backend

/// @brief HeFFTe cuda backend type
typedef uint8_t afft_heffte_cuda_Backend;

/// @brief HeFFTe gpu backend enumeration
#define afft_heffte_gpu_Backend_cufft (afft_heffte_cuda_Backend)0 ///< cuFFT backend

/// @brief HeFFTe hip backend type
typedef uint8_t afft_heffte_hip_Backend;

/// @brief HeFFTe hip backend enumeration
#define afft_heffte_hip_Backend_rocfft (afft_heffte_hip_Backend)0 ///< rocFFT backend

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
/// @brief Backend parameters for single process cpu target
typedef struct afft_cpu_BackendParameters afft_cpu_BackendParameters;

/// @brief Backend parameters for single process cuda target
typedef struct afft_cuda_BackendParameters afft_cuda_BackendParameters;

/// @brief Backend parameters for single process hip target
typedef struct afft_hip_BackendParameters afft_hip_BackendParameters;

/// @brief Backend parameters for single process opencl target
typedef struct afft_opencl_BackendParameters afft_opencl_BackendParameters;

/// @brief Backend parameters for single process openmp target
typedef struct afft_openmp_BackendParameters afft_openmp_BackendParameters;

/// @brief Backend parameters for cpu target
struct afft_cpu_BackendParameters
{
  afft_SelectStrategy       strategy;  ///< Select strategy
  afft_Workspace            workspace; ///< Workspace
  afft_BackendMask          mask;      ///< Backend mask
  size_t                    orderSize; ///< Number of backends in the order
  const afft_Backend*       order;     ///< Order of the backends
  afft_fftw3_cpu_Parameters fftw3;     ///< FFTW3 parameters
};

/// @brief Backend parameters for cuda target
struct afft_cuda_BackendParameters
{
  afft_SelectStrategy        strategy;  ///< Select strategy
  afft_Workspace             workspace; ///< Workspace
  afft_BackendMask           mask;      ///< Backend mask
  size_t                     orderSize; ///< Number of backends in the order
  const afft_Backend*        order;     ///< Order of the backends
  afft_cufft_cuda_Parameters cufft;     ///< cuFFT parameters
};

/// @brief Backend parameters for hip target
struct afft_hip_BackendParameters
{
  afft_SelectStrategy        strategy;  ///< Select strategy
  afft_Workspace             workspace; ///< Workspace
  afft_BackendMask           mask;      ///< Backend mask
  size_t                     orderSize; ///< Number of backends in the order
  const afft_Backend*        order;     ///< Order of the backends
};

/// @brief Backend parameters for opencl target
struct afft_opencl_BackendParameters
{
  afft_SelectStrategy          strategy;  ///< Select strategy
  afft_Workspace               workspace; ///< Workspace
  afft_BackendMask             mask;      ///< Backend mask
  size_t                       orderSize; ///< Number of backends in the order
  const afft_Backend*          order;     ///< Order of the backends
  afft_clfft_opencl_Parameters clfft;     ///< clFFT parameters
};

/// @brief Backend parameters for openmp target
struct afft_openmp_BackendParameters
{
  afft_SelectStrategy strategy;  ///< Select strategy
  afft_Workspace      workspace; ///< Workspace
  afft_BackendMask    mask;      ///< Backend mask
  size_t              orderSize; ///< Number of backends in the order
  const afft_Backend* order;     ///< Order of the backends
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
  afft_Workspace                 workspace; ///< Workspace
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
  afft_Workspace                  workspace; ///< Workspace
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
  afft_Workspace                 workspace; ///< Workspace
  afft_BackendMask               mask;      ///< Backend mask
  size_t                         orderSize; ///< Number of backends in the order
  const afft_Backend*            order;     ///< Order of the backends
  afft_heffte_mpi_hip_Parameters heffte;    ///< HeFFTe parameters
};

/// @brief Backend parameters for cuda target
struct afft_mpi_opencl_BackendParameters
{
  afft_SelectStrategy          strategy;  ///< Select strategy
  afft_Workspace               workspace; ///< Workspace
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
 * @param[in] backend Backend
 * @return Name of the backend
 */
const char* afft_getBackendName(afft_Backend backend);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_BACKEND_H */
