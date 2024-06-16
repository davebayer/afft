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

#ifndef AFFT_ARCHITECTURE_H
#ifndef AFFT_ARCHITECTURE_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Memory block structure
typedef struct
{
  const size_t* starts;  ///< Starts of the memory block
  const size_t* sizes;   ///< Sizes of the memory block
  const size_t* strides; ///< Strides of the memory block
} afft_MemoryBlock;

/**********************************************************************************************************************/
// Spst architectures
/**********************************************************************************************************************/
/// @brief Memory layout structure for spst distribution
typedef struct
{
  const size_t* srcStrides; ///< Stride of the source data
  const size_t* dstStrides; ///< Stride of the destination data
} afft_spst_MemoryLayout;

/// @brief CPU parameters structure for spst architecture
typedef struct
{
  afft_spst_MemoryLayout memoryLayout;   ///< Memory layout
  afft_ComplexFormat     complexFormat;  ///< Complex format
  bool                   preserveSource; ///< Preserve source flag
  afft_Alignment         alignment;      ///< Alignment
  unsigned               threadLimit;    ///< Thread limit
} afft_spst_cpu_Parameters;

/// @brief GPU parameters structure for spst architecture
typedef struct
{
  afft_spst_MemoryLayout memoryLayout;         ///< Memory layout
  afft_ComplexFormat     complexFormat;        ///< Complex format
  bool                   preserveSource;       ///< Preserve source flag
  bool                   useExternalWorkspace; ///< Use external workspace flag
#if AFFT_GPU_BACKEND_IS(CUDA)
  int                    device;               ///< CUDA device
#elif AFFT_GPU_BACKEND_IS(HIP)
  int                    device;               ///< HIP device
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_context             context;              ///< OpenCL context
  cl_device_id           device;               ///< OpenCL device
#endif
} afft_spst_gpu_Parameters;

/// @brief CPU execution parameters structure for spst architecture
typedef struct
{
  uint8_t _dummy; ///< Dummy field to avoid empty struct
} afft_spst_cpu_ExecutionParameters;

/// @brief GPU execution parameters structure for spst architecture
typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t     stream;       ///< CUDA stream
  void*            workspace;    ///< Workspace
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t      stream;       ///< HIP stream
  void*            workspace;    ///< Workspace
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_command_queue commandQueue; ///< OpenCL command queue
  cl_mem           workspace;    ///< Workspace
#else
  uint8_t _dummy;                ///< Dummy field to avoid empty struct
#endif
} afft_spst_gpu_ExecutionParameters;

// Make spst architecture default
typedef afft_spst_cpu_Parameters afft_cpu_Parameters;
typedef afft_spst_gpu_Parameters afft_gpu_Parameters;
typedef afft_spst_cpu_ExecutionParameters afft_cpu_ExecutionParameters;
typedef afft_spst_gpu_ExecutionParameters afft_gpu_ExecutionParameters;

/**********************************************************************************************************************/
// Spmt architectures
/**********************************************************************************************************************/
/// @brief Memory layout structure for spmt distribution
typedef struct
{
  const afft_MemoryBlock* srcBlocks;    ///< Source memory blocks
  const afft_MemoryBlock* dstBlocks;    ///< Destination memory blocks
  const size_t*           srcAxesOrder; ///< Order of the source axes
  const size_t*           dstAxesOrder; ///< Order of the destination axes
} afft_spmt_MemoryLayout;

/// @brief GPU parameters structure for spmt architecture
typedef struct
{
  afft_spmt_MemoryLayout memoryLayout;         ///< Memory layout
  afft_ComplexFormat     complexFormat;        ///< Complex format
  bool                   preserveSource;       ///< Preserve source flag
  bool                   useExternalWorkspace; ///< Use external workspace flag
#if AFFT_GPU_BACKEND_IS(CUDA)
  size_t                 deviceCount;          ///< Number of CUDA devices
  const int*             devices;              ///< CUDA devices
#elif AFFT_GPU_BACKEND_IS(HIP)
  size_t                 deviceCount;          ///< Number of HIP devices
  const int*             devices;              ///< HIP devices
#endif
} afft_spmt_gpu_Parameters;

/// @brief GPU execution parameters structure for spmt architecture
typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t stream;    ///< CUDA stream
  void* const* workspace; ///< Workspace
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t  stream;    ///< HIP stream
  void* const* workspace; ///< Workspace
#else
  uint8_t _dummy;         ///< Dummy field to avoid empty struct
#endif
} afft_spmt_gpu_ExecutionParameters;

/**********************************************************************************************************************/
// Mpst architectures
/**********************************************************************************************************************/
/// @brief Memory layout structure for mpst distribution
typedef struct
{
  afft_MemoryBlock srcBlock;     ///< Source memory block
  afft_MemoryBlock dstBlock;     ///< Destination memory block
  const size_t*    srcAxesOrder; ///< Order of the source axes
  const size_t*    dstAxesOrder; ///< Order of the destination axes
} afft_mpst_MemoryLayout;

/// @brief CPU parameters structure for mpst architecture
typedef struct
{
  afft_mpst_MemoryLayout memoryLayout;         ///< Memory layout
  afft_ComplexFormat     complexFormat;        ///< Complex format
  bool                   preserveSource;       ///< Preserve source flag
  bool                   useExternalWorkspace; ///< Use external workspace flag
#if AFFT_MP_BACKEND_IS(MPI)
  MPI_Comm               communicator;         ///< MPI communicator
#endif
  afft_Alignment         alignment;            ///< Alignment
  unsigned               threadLimit;          ///< Thread limit
} afft_mpst_cpu_Parameters;

/// @brief GPU parameters structure for mpst architecture
typedef struct
{
  afft_mpst_MemoryLayout memoryLayout;         ///< Memory layout
  afft_ComplexFormat     complexFormat;        ///< Complex format
  bool                   preserveSource;       ///< Preserve source flag
  bool                   useExternalWorkspace; ///< Use external workspace flag
#if AFFT_MP_BACKEND_IS(MPI)
  MPI_Comm               communicator;         ///< MPI communicator
#endif
#if AFFT_GPU_BACKEND_IS(CUDA)
  int                    device;               ///< CUDA device
#elif AFFT_GPU_BACKEND_IS(HIP)
  int                    device;               ///< HIP device
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_context             context;              ///< OpenCL context
  cl_device_id           device;               ///< OpenCL device
#endif
} afft_mpst_gpu_Parameters;

/// @brief CPU execution parameters structure for mpst architecture
typedef struct
{
  void* workspace; ///< Workspace
} afft_mpst_cpu_ExecutionParameters;

/// @brief GPU execution parameters structure for mpst architecture
typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t     stream;       ///< CUDA stream
  void*            workspace;    ///< Workspace
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t      stream;       ///< HIP stream
  void*            workspace;    ///< Workspace
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_command_queue commandQueue; ///< OpenCL command queue
  cl_mem           workspace;    ///< Workspace
#else
  uint8_t _dummy;                ///< Dummy field to avoid empty struct
#endif
} afft_mpst_gpu_ExecutionParameters;

/**********************************************************************************************************************/
// General architecture
/**********************************************************************************************************************/
/// @brief Architecture parameters structure
typedef struct
{
  union
  {
    afft_spst_cpu_Parameters spstCpu;
    afft_spst_gpu_Parameters spstGpu;
    afft_spmt_gpu_Parameters spmtGpu;
    afft_mpst_cpu_Parameters mpstCpu;
    afft_mpst_gpu_Parameters mpstGpu;
  };
  afft_Target                target;
  afft_Distribution          distribution;
} afft_ArchitectureParameters;

/// @brief Execution parameters structure
typedef struct
{
  union
  {
    afft_spst_cpu_ExecutionParameters spstCpu;
    afft_spst_gpu_ExecutionParameters spstGpu;
    afft_spmt_gpu_ExecutionParameters spmtGpu;
    afft_mpst_cpu_ExecutionParameters mpstCpu;
    afft_mpst_gpu_ExecutionParameters mpstGpu;
  };
  afft_Target                         target;
  afft_Distribution                   distribution;
} afft_ExecutionParameters;

/**********************************************************************************************************************/
// Private functions
/**********************************************************************************************************************/
static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpstCpu(afft_spst_cpu_Parameters params)
{
  afft_ArchitectureParameters result;
  result.spstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpstGpu(afft_spst_gpu_Parameters params)
{
  afft_ArchitectureParameters result;
  result.spstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpmtGpu(afft_spmt_gpu_Parameters params)
{
  afft_ArchitectureParameters result;
  result.spmtGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spmt;

  return result;
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersMpstCpu(afft_mpst_cpu_Parameters params)
{
  afft_ArchitectureParameters result;
  result.mpstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersMpstGpu(afft_mpst_gpu_Parameters params)
{
  afft_ArchitectureParameters result;
  result.mpstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersAny(afft_ArchitectureParameters params)
{
  return params;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpstCpu(afft_spst_cpu_ExecutionParameters params)
{
  afft_ExecutionParameters result;
  result.spstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpstGpu(afft_spst_gpu_ExecutionParameters params)
{
  afft_ExecutionParameters result;
  result.spstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spst;

  return result;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpmtGpu(afft_spmt_gpu_ExecutionParameters params)
{
  afft_ExecutionParameters result;
  result.spmtGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_spmt;

  return result;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersMpstCpu(afft_mpst_cpu_ExecutionParameters params)
{
  afft_ExecutionParameters result;
  result.mpstCpu      = params;
  result.target       = afft_Target_cpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersMpstGpu(afft_mpst_gpu_ExecutionParameters params)
{
  afft_ExecutionParameters result;
  result.mpstGpu      = params;
  result.target       = afft_Target_gpu;
  result.distribution = afft_Distribution_mpst;

  return result;
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersAny(afft_ExecutionParameters params)
{
  return params;
}

/**********************************************************************************************************************/
// Public functions
/**********************************************************************************************************************/
#ifdef __cplusplus
} // extern "C"

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_spst_cpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpstCpu(params);
}

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_spst_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpstGpu(params);
}

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_spmt_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpmtGpu(params);
}

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_mpst_cpu_Parameters params)
{
  return _afft_makeArchitectureParametersMpstCpu(params);
}

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_mpst_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersMpstGpu(params);
}

/**
 * @brief Make architecture parameters
 * @param params Architecture parameters
 * @return Architecture parameters
 */
static inline afft_ArchitectureParameters afft_makeArchitectureParameters(afft_ArchitectureParameters params)
{
  return _afft_makeArchitectureParametersAny(params);
}

extern "C"
{
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  /**
   * @brief Make architecture parameters
   * @param params Architecture parameters
   * @return Architecture parameters
   */
# define afft_makeArchitectureParameters(params) _Generic((params), \
    afft_spst_cpu_Parameters:    _afft_makeArchitectureParametersSpstCpu, \
    afft_spst_gpu_Parameters:    _afft_makeArchitectureParametersSpstGpu, \
    afft_spmt_gpu_Parameters:    _afft_makeArchitectureParametersSpmtGpu, \
    afft_mpst_cpu_Parameters:    _afft_makeArchitectureParametersMpstCpu, \
    afft_mpst_gpu_Parameters:    _afft_makeArchitectureParametersMpstGpu, \
    afft_ArchitectureParameters: _afft_makeArchitectureParametersAny)(params)
#endif

#ifdef __cplusplus
} // extern "C"

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_spst_cpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpstCpu(params);
}

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_spst_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpstGpu(params);
}

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_spmt_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpmtGpu(params);
}

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_mpst_cpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersMpstCpu(params);
}

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_mpst_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersMpstGpu(params);
}

/**
 * @brief Make execution parameters
 * @param params Execution parameters
 * @return Execution parameters
 */
static inline afft_ExecutionParameters afft_makeExecutionParameters(afft_ExecutionParameters params)
{
  return _afft_makeExecutionParametersAny(params);
}

extern "C"
{
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  /**
   * @brief Make execution parameters
   * @param params Execution parameters
   * @return Execution parameters
   */
# define afft_makeExecutionParameters(params) _Generic((params), \
    afft_spst_cpu_ExecutionParameters: _afft_makeExecutionParametersSpstCpu, \
    afft_spst_gpu_ExecutionParameters: _afft_makeExecutionParametersSpstGpu, \
    afft_spmt_gpu_ExecutionParameters: _afft_makeExecutionParametersSpmtGpu, \
    afft_mpst_cpu_ExecutionParameters: _afft_makeExecutionParametersMpstCpu, \
    afft_mpst_gpu_ExecutionParameters: _afft_makeExecutionParametersMpstGpu, \
    afft_ExecutionParameters:          _afft_makeExecutionParametersAny)(params)
#endif

#ifdef __cplusplus
}
#endif

#endif /* AFFT_ARCHITECTURE_H */
