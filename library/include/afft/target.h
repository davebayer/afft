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

#ifndef AFFT_TARGET_H
#define AFFT_TARGET_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "error.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Target type
typedef uint8_t afft_Target;

/// @brief Target enumeration
#define afft_Target_cpu    (afft_Target)0 ///< CPU
#define afft_Target_cuda   (afft_Target)1 ///< CUDA
#define afft_Target_hip    (afft_Target)2 ///< HIP
#define afft_Target_opencl (afft_Target)3 ///< OpenCL
#define afft_Target_openmp (afft_Target)4 ///< OpenMP

/// @brief CPU parameters
typedef struct afft_cpu_Parameters afft_cpu_Parameters;

/// @brief CPU execution parameters
typedef struct afft_cpu_ExecutionParameters afft_cpu_ExecutionParameters;

/// @brief CUDA parameters
typedef struct afft_cuda_Parameters afft_cuda_Parameters;

/// @brief CUDA execution parameters
typedef struct afft_cuda_ExecutionParameters afft_cuda_ExecutionParameters;

/// @brief HIP parameters
typedef struct afft_hip_Parameters afft_hip_Parameters;

/// @brief HIP execution parameters
typedef struct afft_hip_ExecutionParameters afft_hip_ExecutionParameters;

/// @brief OpenCL parameters
typedef struct afft_opencl_Parameters afft_opencl_Parameters;

/// @brief OpenCL execution parameters
typedef struct afft_opencl_ExecutionParameters afft_opencl_ExecutionParameters;

/// @brief OpenMP parameters
typedef struct afft_openmp_Parameters afft_openmp_Parameters;

/// @brief OpenMP execution parameters
typedef struct afft_openmp_ExecutionParameters afft_openmp_ExecutionParameters;

#ifdef AFFT_ENABLE_CPU
/// @brief CPU parameters
struct afft_cpu_Parameters
{
  unsigned threadLimit; ///< Thread limit
};

/// @brief CPU execution parameters
struct afft_cpu_ExecutionParameters
{
  void* externalWorkspace; ///< External workspace, if afft_Workspace_external is used
};
#endif

#ifdef AFFT_ENABLE_CUDA
/// @brief CUDA parameters
struct afft_cuda_Parameters
{
  size_t     targetCount; ///< Target count
  const int* devices;     ///< Device ids
};

/// @brief CUDA execution parameters
struct afft_cuda_ExecutionParameters
{
  cudaStream_t stream;             ///< CUDA stream
  void* const* externalWorkspaces; ///< External workspace, if afft_Workspace_external is used
};
#endif

#ifdef AFFT_ENABLE_HIP
/// @brief HIP parameters
struct afft_hip_Parameters
{
  size_t     targetCount; ///< Target count
  const int* devices;     ///< Device ids
};

/// @brief HIP execution parameters
struct afft_hip_ExecutionParameters
{
  hipStream_t  stream;             ///< HIP stream
  void* const* externalWorkspaces; ///< External workspace, if afft_Workspace_external is used
};
#endif

#ifdef AFFT_ENABLE_OPENCL
/// @brief OpenCL parameters
struct afft_opencl_Parameters
{
  size_t              targetCount; ///< Target count
  cl_context          context;     ///< OpenCL context
  const cl_device_id* devices;     ///< Device ids
};

/// @brief OpenCL execution parameters
struct afft_opencl_ExecutionParameters
{
  cl_command_queue queue;              ///< OpenCL command queue
  const cl_mem*    externalWorkspaces; ///< External workspace, if afft_Workspace_external is used
};
#endif

#ifdef AFFT_ENABLE_OPENMP
/// @brief OpenMP parameters
struct afft_openmp_Parameters
{
  int device; ///< OpenMP device
};

/// @brief OpenMP execution parameters
struct afft_openmp_ExecutionParameters
{
  bool nowait; ///< Nowait
};
#endif

/// @brief Target parameters variant
typedef struct afft_TargetParametersVariant afft_TargetParametersVariant;

/// @brief Target parameters variant structure
struct afft_TargetParametersVariant
{
  afft_Target target;              ///< Target type
  union
  {
#ifdef AFFT_ENABLE_CPU
    afft_cpu_Parameters cpu;       ///< CPU parameters
#endif
#ifdef AFFT_ENABLE_CUDA
    afft_cuda_Parameters cuda;     ///< CUDA parameters
#endif
#ifdef AFFT_ENABLE_HIP
    afft_hip_Parameters hip;       ///< HIP parameters
#endif
#ifdef AFFT_ENABLE_OPENCL
    afft_opencl_Parameters opencl; ///< OpenCL parameters
#endif
#ifdef AFFT_ENABLE_OPENMP
    afft_openmp_Parameters openmp; ///< OpenMP parameters
#endif
    char                   _dummy; ///< Dummy member to ensure that the union is not empty
  };
};

#ifdef __cplusplus
}
#endif

#endif /* AFFT_TARGET_H */
