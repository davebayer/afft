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

#ifndef AFFT_H
#define AFFT_H

#ifdef __STDC_VERSION__
# if __STDC_VERSION__ < 201112L
#   error "C11 or later is required"
# endif
#endif

#include <stdbool.h>
#include <float.h>
#ifndef __STDC_NO_COMPLEX__
# include <complex.h>
#endif

#include "config.hpp"

#if AFFT_GPU_BACKEND_IS(CUDA)
# include <cuda_runtime.h>
#elif AFFT_GPU_BACKEND_IS(HIP)
# include <hip/hip_runtime.h>
#elif AFFT_GPU_BACKEND_IS(OPENCL)
# if defined(__APPLE__) || defined(__MACOSX)
#   include <OpenCL/cl.h>
# else
#   include <CL/cl.h>
# endif
#endif

#if AFFT_MP_BACKEND_IS(MPI)
# include <mpi.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**********************************************************************************************************************/
// Error
/**********************************************************************************************************************/

/// @brief Error enumeration
typedef enum
{
  afft_Error_success,
  afft_Error_internal,
  afft_Error_invalidPlan,
  afft_Error_invalidArgument,

  afft_Error_invalidPrecision,
  afft_Error_invalidComplexity,
  afft_Error_invalidComplexFormat,
  afft_Error_invalidDirection,
  afft_Error_invalidPlacement,
  afft_Error_invalidTransform,
  afft_Error_invalidTarget,
  afft_Error_invalidDistribution,
  afft_Error_invalidNormalization,
} afft_Error;

/**********************************************************************************************************************/
// Common
/**********************************************************************************************************************/

/// @brief Precision enumeration
typedef enum
{
  _afft_Precision_unknown = -1, ///< Unknown precision, only for internal use
  afft_Precision_bf16,          ///< Google Brain's brain floating-point format
  afft_Precision_f16,           ///< IEEE 754 half-precision binary floating-point format
  afft_Precision_f32,           ///< IEEE 754 single-precision binary floating-point format
  afft_Precision_f64,           ///< IEEE 754 double-precision binary floating-point format
  afft_Precision_f80,           ///< x86 80-bit extended precision format
  afft_Precision_f64f64,        ///< double double precision (f128 simulated with two f64)
  afft_Precision_f128,          ///< IEEE 754 quadruple-precision binary floating-point format

  afft_Precision_float        = afft_Precision_f32,
  afft_Precision_double       = afft_Precision_f64,
# if LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024 && LDBL_MIN_EXP == -1021
  afft_Precision_float        = afft_Precision_f64,    ///< Precision of long double
# elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
  afft_Precision_longDouble   = afft_Precision_f80,    ///< Precision of long double
# elif (LDBL_MANT_DIG >=   105 && LDBL_MANT_DIG <=   107) && \
       (LDBL_MAX_EXP  >=  1023 && LDBL_MAX_EXP  <=  1025) && \
       (LDBL_MIN_EXP  >= -1022 && LDBL_MIN_EXP  <= -1020)
  afft_Precision_longDouble   = afft_Precision_f64f64, ///< Precision of long double
# elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
  afft_Precision_longDouble   = afft_Precision_f128,   ///< Precision of long double
# else
#   error "Unrecognized long double format"
# endif
  afft_Precision_doubleDouble = afft_Precision_f64f64, ///< Precision of double double
  afft_Precision_quad         = afft_Precision_f128,   ///< Precision of quad
} afft_Precision;

/// @brief Alignment type
typedef size_t afft_Alignment;

/// @brief Complexity enumeration
typedef enum
{
  _afft_Complexity_unknown = -1, ///< Unknown complexity, only for internal use
  afft_Complexity_real,          ///< Real
  afft_Complexity_complex,       ///< Complex
} afft_Complexity;

/// @brief Complex format enumeration
typedef enum
{
  afft_ComplexFormat_interleaved, ///< Interleaved
  afft_ComplexFormat_planar       ///< Planar
} afft_ComplexFormat;

/// @brief Direction enumeration
typedef enum
{
  afft_Direction_forward, ///< Forward
  afft_Direction_inverse, ///< Inverse

  afft_Direction_backward = afft_Direction_inverse, ///< Alias for inverse
} afft_Direction;

/// @brief Placement enumeration
typedef enum
{
  afft_Placement_inPlace,    ///< In-place
  afft_Placement_outOfPlace, ///< Out-of-place

  afft_Placement_notInPlace = afft_Placement_outOfPlace, ///< Alias for outOfPlace
} afft_Placement;

/// @brief Transform enumeration
typedef enum
{
  _afft_Transform_unknown = -1, ///< Unknown transform, only for internal use
  afft_Transform_dft,           ///< Discrete Fourier Transform
  afft_Transform_dht,           ///< Discrete Hartley Transform
  afft_Transform_dtt,           ///< Discrete Trigonometric Transform
} afft_Transform;

/// @brief Target enumeration
typedef enum
{
  _afft_Target_unknown = -1, ///< Unknown target, only for internal use
  afft_Target_cpu,           ///< CPU
  afft_Target_gpu,           ///< GPU
} afft_Target;

/// @brief Distribution enumeration
typedef enum
{
  _afft_Distribution_unknown = -1, ///< Unknown distribution, only for internal use
  afft_Distribution_spst,          ///< Single process, single target
  afft_Distribution_spmt,          ///< Single process, multiple targets
  afft_Distribution_mpst,          ///< Multiple processes, single target

  afft_Distribution_single = afft_Distribution_spst, ///< Alias for single process, single target
  afft_Distribution_multi  = afft_Distribution_spmt, ///< Alias for single process, multiple targets
  afft_Distribution_mpi    = afft_Distribution_mpst, ///< Alias for multiple processes, single target
} afft_Distribution;

/// @brief Normalization enumeration
typedef enum
{
  afft_Normalization_none,       ///< No normalization
  afft_Normalization_orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
  afft_Normalization_unitary,    ///< 1/N normalization applied to inverse transform
} afft_Normalization;

/// @brief Precision triad structure
typedef struct
{
  afft_Precision execution;   ///< Precision of the execution
  afft_Precision source;      ///< Precision of the source data
  afft_Precision destination; ///< Precision of the destination data
} afft_PrecisionTriad;

/**********************************************************************************************************************/
// Backend
/**********************************************************************************************************************/

/// @brief Backend enumeration
typedef enum
{
  afft_Backend_clfft,     ///< clFFT
  afft_Backend_cufft,     ///< cuFFT
  afft_Backend_fftw3,     ///< FFTW3
  afft_Backend_heffte,    ///< HeFFTe
  afft_Backend_hipfft,    ///< hipFFT
  afft_Backend_mkl,       ///< Intel MKL
  afft_Backend_pocketfft, ///< PocketFFT
  afft_Backend_rocfft,    ///< rocFFT
  afft_Backend_vkfft,     ///< VkFFT
  _afft_Backend_count,    ///< number of backends, do not use, only for internal purposes
} afft_Backend;

/// @brief Backend mask enumeration
typedef enum
{
  afft_BackendMask_empty = 0,       ///< Empty mask
  afft_BackendMask_all   = INT_MAX, ///< All backends
} afft_BackendMask;

_Static_assert(sizeof(int) * CHAR_BIT >= _afft_Backend_count,
               "afft_BackendMask does not have sufficient size to store all Backend values");

/// @brief Select strategy enumeration
typedef enum
{
  afft_SelectStrategy_first, ///< Select the first available backend
  afft_SelectStrategy_best,  ///< Select the best available backend
} afft_SelectStrategy;

/// @brief clFFT backend parameters for spst gpu architecture
typedef struct
{
  bool useFastMath;
} afft_spst_gpu_clfft_Parameters;

/// @brief cuFFT workspace policy enumeration
typedef enum
{
  afft_cufft_WorkspacePolicy_performance, ///< Use the workspace for performance
  afft_cufft_WorkspacePolicy_minimal,     ///< Use the minimal workspace
  afft_cufft_WorkspacePolicy_user,        ///< Use the user-defined workspace size
} afft_cufft_WorkspacePolicy;

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

/// @brief FFTW3 planner flag enumeration
typedef enum
{
  afft_fftw3_PlannerFlag_estimate,        ///< Estimate plan flag
  afft_fftw3_PlannerFlag_measure,         ///< Measure plan flag
  afft_fftw3_PlannerFlag_patient,         ///< Patient plan flag
  afft_fftw3_PlannerFlag_exhaustive       ///< Exhaustive planner flag
  afft_fftw3_PlannerFlag_estimatePatient, ///< Estimate and patient plan flag
} afft_fftw3_PlannerFlag;

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

/// @brief HeFFTe backend enumeration
typedef enum
{
  afft_heffte_Backend_cufft,  ///< cuFFT backend
  afft_heffte_Backend_fftw3,  ///< FFTW3 backend
  afft_heffte_Backend_mkl,    ///< MKL backend
  afft_heffte_Backend_rocfft, ///< rocFFT backend
} afft_heffte_Backend;

/// @brief HeFFTe backend parameters for mpst cpu architecture
typedef struct
{
  afft_heffte_Backend backend;     ///< HeFFTe backend
  bool                useReorder;  ///< Use reorder flag
  bool                useAllToAll; ///< Use all-to-all flag
  bool                usePencils;  ///< Use pencils flag
} afft_mpst_cpu_heffte_Parameters;

/// @brief HeFFTe backend parameters for mpst gpu architecture
typedef struct
{
  afft_heffte_Backend backend;     ///< HeFFTe backend
  bool                useReorder;  ///< Use reorder flag
  bool                useAllToAll; ///< Use all-to-all flag
  bool                usePencils;  ///< Use pencils flag
} afft_mpst_gpu_heffte_Parameters;

/// @brief Backend parameters for spst cpu architecture
typedef struct
{
  afft_SelectStrategy            selectStrategy; ///< Select strategy
  afft_BackendMask               backendMask;    ///< Backend mask
  size_t                         orderCount;     ///< Number of backends in the order
  const afft_Backend*            order;          ///< Order of the backends
  afft_spst_cpu_fftw3_Parameters fftw3;          ///< FFTW3 parameters
} afft_spst_cpu_BackendParameters;

/// @brief Backend parameters for spst gpu architecture
typedef struct
{
  afft_SelectStrategy            selectStrategy; ///< Select strategy
  afft_BackendMask               backendMask;    ///< Backend mask
  size_t                         orderCount;     ///< Number of backends in the order
  const afft_Backend*            order;          ///< Order of the backends
  afft_spst_gpu_clfft_Parameters clfft;          ///< clFFT parameters
  afft_spst_gpu_cufft_Parameters cufft;          ///< cuFFT parameters
} afft_spst_gpu_BackendParameters;

/// @brief Backend parameters for spmt gpu architecture
typedef struct
{
  afft_SelectStrategy            selectStrategy; ///< Select strategy
  afft_BackendMask               backendMask;    ///< Backend mask
  size_t                         orderCount;     ///< Number of backends in the order
  const afft_Backend*            order;          ///< Order of the backends
  afft_spmt_gpu_cufft_Parameters cufft;          ///< cuFFT parameters
} afft_spmt_gpu_BackendParameters;

/// @brief Backend parameters for mpst cpu architecture
typedef struct
{
  afft_SelectStrategy             selectStrategy; ///< Select strategy
  afft_BackendMask                backendMask;    ///< Backend mask
  size_t                          orderCount;     ///< Number of backends in the order
  const afft_Backend*             order;          ///< Order of the backends
  afft_mpst_cpu_fftw3_Parameters  fftw3;          ///< FFTW3 parameters
  afft_mpst_cpu_heffte_Parameters heffte;         ///< HeFFTe parameters
} afft_mpst_cpu_BackendParameters;

/// @brief Backend parameters for mpst gpu architecture
typedef struct
{
  afft_SelectStrategy             selectStrategy; ///< Select strategy
  afft_BackendMask                backendMask;    ///< Backend mask
  size_t                          orderCount;     ///< Number of backends in the order
  const afft_Backend*             order;          ///< Order of the backends
  afft_mpst_gpu_cufft_Parameters  cufft;          ///< cuFFT parameters
  afft_mpst_gpu_heffte_Parameters heffte;         ///< HeFFTe parameters
} afft_mpst_gpu_BackendParameters;

afft_spst_cpu_BackendParameters afft_spst_cpu_makeDefaultBackendParameters();
afft_spst_gpu_BackendParameters afft_spst_gpu_makeDefaultBackendParameters();
afft_spmt_gpu_BackendParameters afft_spmt_gpu_makeDefaultBackendParameters();
afft_mpst_cpu_BackendParameters afft_mpst_cpu_makeDefaultBackendParameters();
afft_mpst_gpu_BackendParameters afft_mpst_gpu_makeDefaultBackendParameters();

typedef afft_gpu_clfft_Parameters afft_spst_gpu_clfft_Parameters;
typedef afft_gpu_cufft_Parameters afft_spst_gpu_cufft_Parameters;
typedef afft_cpu_fftw3_Parameters afft_spst_cpu_fftw3_Parameters;

typedef afft_spst_cpu_BackendParameters afft_cpu_BackendParameters;
typedef afft_spst_gpu_BackendParameters afft_gpu_BackendParameters;

static inline afft_cpu_BackendParameters afft_cpu_makeDefaultBackendParameters()
{
  return afft_spst_cpu_makeDefaultBackendParameters();
}

static inline afft_gpu_BackendParameters afft_gpu_makeDefaultBackendParameters()
{
  return afft_spst_gpu_makeDefaultBackendParameters();
}

/// @brief Feedback structure
typedef struct
{
  afft_Backend backend;      ///< Backend
  char*        message;      ///< Message from the backend
  double       measuredTime; ///< Measured time in seconds
} afft_Feedback;

const char* getBackendName(afft_Backend backend);

/**********************************************************************************************************************/
// Distribution
/**********************************************************************************************************************/

/// @brief Memory block structure
typedef struct
{
  const size_t* starts;  ///< Starts of the memory block
  const size_t* sizes;   ///< Sizes of the memory block
  const size_t* strides; ///< Strides of the memory block
} afft_MemoryBlock;

/// @brief Memory layout structure for spst distribution
typedef struct
{
  const size_t* srcStrides; ///< Stride of the source data
  const size_t* dstStrides; ///< Stride of the destination data
} afft_spst_MemoryLayout;

/// @brief Memory layout structure for spmt distribution
typedef struct
{
  const afft_MemoryBlock* srcBlocks;    ///< Source memory blocks
  const afft_MemoryBlock* dstBlocks;    ///< Destination memory blocks
  const size_t*           srcAxesOrder; ///< Order of the source axes
  const size_t*           dstAxesOrder; ///< Order of the destination axes
} afft_spmt_MemoryLayout;

/// @brief Memory layout structure for mpst distribution
typedef struct
{
  afft_MemoryBlock srcBlock;     ///< Source memory block
  afft_MemoryBlock dstBlock;     ///< Destination memory block
  const size_t*    srcAxesOrder; ///< Order of the source axes
  const size_t*    dstAxesOrder; ///< Order of the destination axes
} afft_mpst_MemoryLayout;

/**********************************************************************************************************************/
// Transforms
/**********************************************************************************************************************/

/// @brief DFT transform type enumeration
typedef enum
{
  afft_dft_Type_complexToComplex, ///< Complex-to-complex transform
  afft_dft_Type_realToComplex,    ///< Real-to-complex transform
  afft_dft_Type_complexToReal,    ///< Complex-to-real transform

  afft_dft_Type_c2c = afft_dft_Type_complexToComplex, ///< Alias for complex-to-complex transform
  afft_dft_Type_r2c = afft_dft_Type_realToComplex,    ///< Alias for real-to-complex transform
  afft_dft_Type_c2r = afft_dft_Type_complexToReal,    ///< Alias for complex-to-real transform
} afft_dft_Type;

/// @brief DFT parameters structure
typedef struct
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const size_t*       shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const size_t*       axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dft_Type       type;          ///< Type of the transform
} afft_dft_Parameters;

/// @brief DHT transform type enumeration
typedef enum
{
  afft_dht_Type_separable, ///< Separable DHT, computes the DHT along each axis independently
} afft_dht_Type;

/// @brief DHT parameters structure
typedef struct
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const size_t*       shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const size_t*       axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dht_Type       type;          ///< Type of the transform
} afft_dht_Parameters;

/// @brief DTT transform type enumeration
typedef enum
{
  afft_dtt_Type_dct1, ///< Discrete Cosine Transform type I
  afft_dtt_Type_dct2, ///< Discrete Cosine Transform type II
  afft_dtt_Type_dct3, ///< Discrete Cosine Transform type III
  afft_dtt_Type_dct4, ///< Discrete Cosine Transform type IV

  afft_dtt_Type_dst1, ///< Discrete Sine Transform type I
  afft_dtt_Type_dst2, ///< Discrete Sine Transform type II
  afft_dtt_Type_dst3, ///< Discrete Sine Transform type III
  afft_dtt_Type_dst4, ///< Discrete Sine Transform type IV

  afft_dtt_Type_dct = afft_dtt_Type_dct2, ///< default DCT type
  afft_dtt_Type_dst = afft_dtt_Type_dst2, ///< default DST type
} afft_dtt_Type;

/// @brief DTT parameters structure
typedef struct
{
  afft_Direction       direction;     ///< Direction of the transform
  afft_PrecisionTriad  precision;     ///< Precision triad
  size_t               shapeRank;     ///< Rank of the shape
  const size_t*        shape;         ///< Shape of the transform
  size_t               axesRank;      ///< Rank of the axes
  const size_t*        axes;          ///< Axes of the transform
  afft_Normalization   normalization; ///< Normalization
  afft_Placement       placement;     ///< Placement of the transform
  const afft_dtt_Type* types;         ///< Types of the transform
} afft_dtt_Parameters;

/**********************************************************************************************************************/
// Targets
/**********************************************************************************************************************/

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

typedef afft_spst_cpu_Parameters afft_cpu_Parameters;
typedef afft_spst_gpu_Parameters afft_gpu_Parameters;

/// @brief CPU execution parameters structure for spst architecture
typedef struct
{
  // no parameters
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
#endif
} afft_spst_gpu_ExecutionParameters;

/// @brief GPU execution parameters structure for spmt architecture
typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t stream;    ///< CUDA stream
  void* const* workspace; ///< Workspace
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t  stream;    ///< HIP stream
  void* const* workspace; ///< Workspace
#endif
} afft_spmt_gpu_ExecutionParameters;

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
#endif
} afft_mpst_gpu_ExecutionParameters;

typedef afft_spst_cpu_ExecutionParameters afft_cpu_ExecutionParameters;
typedef afft_spst_gpu_ExecutionParameters afft_gpu_ExecutionParameters;

/**********************************************************************************************************************/
// Plan
/**********************************************************************************************************************/

/// @brief Opaque plan structure
typedef struct _afft_Plan afft_Plan;

/// @brief Transform parameters structure
typedef struct
{
  union
  {
    afft_dft_Parameters dft;
    afft_dht_Parameters dht;
    afft_dtt_Parameters dtt;
  };
  afft_Transform        transform;
} afft_TransformParameters;

static inline afft_TransformParameters _afft_makeTransformParametersDft(afft_dft_Parameters params)
{
  return (afft_TransformParameters){.dft = params, .transform = afft_Transform_dft};
}

static inline afft_TransformParameters _afft_makeTransformParametersDht(afft_dht_Parameters params)
{
  return (afft_TransformParameters){.dht = params, .transform = afft_Transform_dht};
}

static inline afft_TransformParameters _afft_makeTransformParametersDtt(afft_dtt_Parameters params)
{
  return (afft_TransformParameters){.dtt = params, .transform = afft_Transform_dtt};
}

static inline afft_TransformParameters _afft_makeTransformParametersAny(afft_TransformParameters params)
{
  return params;
}

#ifndef __cplusplus
# define _afft_makeTransformParameters(params) _Generic((params), \
    afft_dft_Parameters:      _afft_makeTransformParametersDft, \
    afft_dht_Parameters:      _afft_makeTransformParametersDht, \
    afft_dtt_Parameters:      _afft_makeTransformParametersDtt, \
    afft_TransformParameters: _afft_makeTransformParametersAny)(params)
#else
static inline _afft_makeTransformParameters(afft_dft_Parameters params)
{
  return _afft_makeTransformParametersDft(params);
}

static inline _afft_makeTransformParameters(afft_dht_Parameters params)
{
  return _afft_makeTransformParametersDht(params);
}

static inline _afft_makeTransformParameters(afft_dtt_Parameters params)
{
  return _afft_makeTransformParametersDtt(params);
}

static inline _afft_makeTransformParameters(afft_TransformParameters params)
{
  return _afft_makeTransformParameters(params);
}
#endif

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

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpstCpu(afft_spst_cpu_Parameters params)
{
  return (afft_ArchitectureParameters){.spstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_spst};
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpstGpu(afft_spst_gpu_Parameters params)
{
  return (afft_ArchitectureParameters){.spstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spst};
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersSpmtGpu(afft_spmt_gpu_Parameters params)
{
  return (afft_ArchitectureParameters){.spmtGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spmt};
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersMpstCpu(afft_mpst_cpu_Parameters params)
{
  return (afft_ArchitectureParameters){.mpstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_mpst};
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersMpstGpu(afft_mpst_gpu_Parameters params)
{
  return (afft_ArchitectureParameters){.mpstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_mpst};
}

static inline afft_ArchitectureParameters _afft_makeArchitectureParametersAny(afft_ArchitectureParameters params)
{
  return params;
}

#ifndef __cplusplus
# define _afft_makeArchitectureParameters(params) _Generic((params), \
    afft_spst_cpu_Parameters:    _afft_makeArchitectureParametersSpstCpu, \
    afft_spst_gpu_Parameters:    _afft_makeArchitectureParametersSpstGpu, \
    afft_spmt_gpu_Parameters:    _afft_makeArchitectureParametersSpmtGpu, \
    afft_mpst_cpu_Parameters:    _afft_makeArchitectureParametersMpstCpu, \
    afft_mpst_gpu_Parameters:    _afft_makeArchitectureParametersMpstGpu, \
    afft_ArchitectureParameters: _afft_makeArchitectureParametersAny)(params)
#else
static inline _afft_makeArchitectureParameters(afft_spst_cpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpstCpu(params);
}

static inline _afft_makeArchitectureParameters(afft_spst_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpstGpu(params);
}

static inline _afft_makeArchitectureParameters(afft_spmt_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersSpmtGpu(params);
}

static inline _afft_makeArchitectureParameters(afft_mpst_cpu_Parameters params)
{
  return _afft_makeArchitectureParametersMpstCpu(params);
}

static inline _afft_makeArchitectureParameters(afft_mpst_gpu_Parameters params)
{
  return _afft_makeArchitectureParametersMpstGpu(params);
}

static inline _afft_makeArchitectureParameters(afft_ArchitectureParameters params)
{
  return _afft_makeArchitectureParametersAny(params);
}
#endif

/// @brief Backend parameters structure
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
} afft_BackendParameters;

static inline afft_BackendParameters _afft_makeBackendParametersSpstCpu(afft_spst_cpu_Parameters params)
{
  return (afft_BackendParameters){.spstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_spst};
}

static inline afft_BackendParameters _afft_makeBackendParametersSpstGpu(afft_spst_gpu_Parameters params)
{
  return (afft_BackendParameters){.spstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spst};
}

static inline afft_BackendParameters _afft_makeBackendParametersSpmtGpu(afft_spmt_gpu_Parameters params)
{
  return (afft_BackendParameters){.spmtGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spmt};
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstCpu(afft_mpst_cpu_Parameters params)
{
  return (afft_BackendParameters){.mpstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_mpst};
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstGpu(afft_mpst_gpu_Parameters params)
{
  return (afft_BackendParameters){.mpstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_mpst};
}

static inline afft_BackendParameters _afft_makeBackendParametersAny(afft_BackendParameters params)
{
  return params;
}

#ifndef __cplusplus
# define _afft_makeBackendParameters(params) _Generic((params), \
    afft_spst_cpu_Parameters: _afft_makeBackendParametersSpstCpu, \
    afft_spst_gpu_Parameters: _afft_makeBackendParametersSpstGpu, \
    afft_spmt_gpu_Parameters: _afft_makeBackendParametersSpmtGpu, \
    afft_mpst_cpu_Parameters: _afft_makeBackendParametersMpstCpu, \
    afft_mpst_gpu_Parameters: _afft_makeBackendParametersMpstGpu, \
    afft_BackendParameters:   _afft_makeBackendParametersAny)(params)
#else
static inline _afft_makeBackendParameters(afft_spst_cpu_Parameters params)
{
  return _afft_makeBackendParametersSpstCpu(params);
}

static inline _afft_makeBackendParameters(afft_spst_gpu_Parameters params)
{
  return _afft_makeBackendParametersSpstGpu(params);
}

static inline _afft_makeBackendParameters(afft_spmt_gpu_Parameters params)
{
  return _afft_makeBackendParametersSpmtGpu(params);
}

static inline _afft_makeBackendParameters(afft_mpst_cpu_Parameters params)
{
  return _afft_makeBackendParametersMpstCpu(params);
}

static inline _afft_makeBackendParameters(afft_mpst_gpu_Parameters params)
{
  return _afft_makeBackendParametersMpstGpu(params);
}

static inline _afft_makeBackendParameters(afft_BackendParameters params)
{
  return _afft_makeBackendParametersAny(params);
}
#endif

#define afft_makePlan(planPtr, transformParams, archParams) \
  _afft_makePlan(planPtr, \
                 _afft_makeTransformParameters(transformParams), \
                 _afft_makeArchitectureParameters(archParams))

#define afft_makePlanWithBackendParameters(planPtr, transformParams, archParams, backendParams) \
  _afft_makePlanWithBackendParameters(planPtr, \
                                      _afft_makeTransformParameters(transformParams), \
                                      _afft_makeArchitectureParameters(archParams), \
                                      _afft_makeBackendParameters(backendParams))

afft_Error afft_Plan_getTransform(const afft_Plan* plan, afft_Transform* transform);

afft_Error afft_Plan_getTarget(const afft_Plan* plan, afft_Target* target);

afft_Error afft_Plan_getTargetCount(const afft_Plan* plan, size_t* targetCount);

afft_Error afft_Plan_getDistribution(const afft_Plan* plan, afft_Distribution* distribution);

afft_Error afft_Plan_getBackend(const afft_Plan* plan, afft_Backend* backend);

afft_Error afft_Plan_getWorkspaceSize(const afft_Plan* plan, size_t* workspaceSize);

void afft_Plan_destroy(afft_Plan* plan);

// typedef struct
// {
//   size_t          ptrCount;
//   union
//   {
//     void*         ptr;
//     void* const*  ptrs;
//   }
//   afft_Precision  precision;
//   afft_Complexity complexity;
//   bool            isConst;
// } _afft_ExecParam;

// TODO:
// #define AFFT_PLANAR_COMPLEX(_type) afft_PlanarComplex_##_type
// #define AFFT_PLANAR_COMPLEX_CONST(_type) afft_PlanarComplex_const_##_type

// #define AFFT_DEFINE_PLANAR_COMPLEX_WITH_NAME(_type, _name)

// #define AFFT_DEFINE_PLANAR_COMPLEX(_type) \
//   typedef struct \
//   { \
//     _type* real; \
//     _type* imag; \
//   } AFFT_PLANAR_COMPLEX(_type); \
//   \
//   typedef struct \
//   { \
//     const _type* real; \
//     const _type* imag; \
//   } AFFT_PLANAR_COMPLEX_CONST(_type);



// #define AFFT_PLANAR_COMPLEX(_name, _type) \
//   typedef struct \
//   { \
//     _type* real; \
//     _type* imag; \
//   } afft_PlanarComplex_##_name; \

// #define AFFT_REAL_TYPE_PROPERTIES(_name, _type, _precision) \
//   AFFT_PLANAR_COMPLEX(_name, _type) \
//   AFFT_PLANAR_COMPLEX(Const##_name, _type) \
//   \
//   static inline _afft_ExecParam _afft_makeExecParam_##_name(_type* ptr) \
//   { \
//     return (_afft_ExecParam){.ptr = ptr, .precision = _precision_, .complexity = afft_Complexity_real, .isConst = false}; \
//   } \
//   \
//   static inline _afft_ExecParam _afft_makeExecParam_Const##_name(const _type* ptr) \
//   { \
//     return (_afft_ExecParam){.ptr = (void**)ptr, .precision = _precision_, .complexity = afft_Complexity_real, .isConst = true}; \
//   } \
//   \
//   static inline _afft_ExecParam _afft_makeExecParam_PlanarComplex_##_name(afft_PlanarComplex_##_name planar) \
//   { \
//     return (_afft_ExecParam){.ptrs = (void* const*)&planar, .precision = _precision_, .complexity = afft_Complexity_complex, .isConst = false}; \
//   } \
//   \
//   static inline _afft_ExecParam _afft_makeExecParam_PlanarComplex_Const##_name(afft_PlanarComplex_Const##name planar) \
//   { \
//     return (_afft_ExecParam){.ptr = (void* const*)&planar, .precision = _precision_, .complexity = afft_Complexity_complex, .isConst = true}; \
//   }

// #define AFFT_COMPLEX_TYPE_PROPERTIES(_name, _type, _precision) \
//   static inline _afft_ExecParam _afft_makeExecParam_##_name(_type* ptr) \
//   { \
//     return (_afft_ExecParam){.ptr = ptr, .precision = _precision_, .complexity = afft_Complexity_complex, .isConst = false}; \
//   } \
//   \
//   static inline _afft_ExecParam _afft_makeExecParam_Const##_name(const _type* ptr) \
//   { \
//     return (_afft_ExecParam){.ptr = (void*)ptr, .precision = _precision_, .complexity = afft_Complexity_complex, .isConst = true}; \
//   }






















typedef struct
{
  void*           ptr;
  afft_Precision  precision;
  afft_Complexity complexity;
  bool            isConst;
} _afft_ExecParam;


typedef struct
{
  const void*       ptr;
  afft_Target       target;
  afft_Distribution distribution;
} _afft_ArchParam;









typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t     stream;
  void*            workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t      stream;
  void*            workspace;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_command_queue commandQueue;
  cl_mem           workspace;
#endif
} afft_spst_gpu_ExecutionParameters;

typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t stream;
  void* const* workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t  stream;
  void* const* workspace;
#endif
} afft_spmt_gpu_ExecutionParameters;

typedef struct
{
  void* workspace;
} afft_mpst_cpu_ExecutionParameters;

typedef struct
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaStream_t     stream;
  void*            workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipStream_t      stream;
  void*            workspace;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cl_command_queue commandQueue;
  cl_mem           workspace;
#endif
} afft_mpst_gpu_ExecutionParameters;

typedef afft_spst_cpu_ExecutionParameters afft_cpu_ExecutionParameters;
typedef afft_spst_gpu_ExecutionParameters afft_gpu_ExecutionParameters;

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
} _afft_ArchParam;

#ifndef __cplusplus
# define _afft_makeArchParam(param) _Generic((param), \
    afft_spst_cpu_Parameters: _afft_makeArchParamSpstCpu, \
    afft_spst_gpu_Parameters: _afft_makeArchParamSpstGpu, \
    afft_spmt_gpu_Parameters: _afft_makeArchParamSpmtGpu, \
    afft_mpst_cpu_Parameters: _afft_makeArchParamMpstCpu, \
    afft_mpst_gpu_Parameters: _afft_makeArchParamMpstGpu)(param)
#else
static inline _afft_makeArchParam(afft_spst_cpu_Parameters param)
{
  return _afft_makeArchParamSpstCpu(param);
}

static inline _afft_makeArchParam(afft_spst_gpu_Parameters param)
{
  return _afft_makeArchParamSpstGpu(param);
}

static inline _afft_makeArchParam(afft_spmt_gpu_Parameters param)
{
  return _afft_makeArchParamSpmtGpu(param);
}

static inline _afft_makeArchParam(afft_mpst_cpu_Parameters param)
{
  return _afft_makeArchParamMpstCpu(param);
}

static inline _afft_makeArchParam(afft_mpst_gpu_Parameters param)
{
  return _afft_makeArchParamMpstGpu(param);
}
#endif

static inline _afft_ArchParam _afft_makeArchParamSpstCpu(afft_spst_cpu_Parameters param)
{
  return (_afft_ArchParam){.spstCpu = param, .target = afft_Target_cpu, .distribution = afft_Distribution_spst};
}

static inline _afft_ArchParam _afft_makeArchParamSpstGpu(afft_spst_gpu_Parameters param)
{
  return (_afft_ArchParam){.spstGpu = param, .target = afft_Target_gpu, .distribution = afft_Distribution_spst};
}

static inline _afft_ArchParam _afft_makeArchParamSpmtGpu(afft_spmt_gpu_Parameters param)
{
  return (_afft_ArchParam){.spmtGpu = param, .target = afft_Target_gpu, .distribution = afft_Distribution_spmt};
}

static inline _afft_ArchParam _afft_makeArchParamMpstCpu(afft_mpst_cpu_Parameters param)
{
  return (_afft_ArchParam){.mpstCpu = param, .target = afft_Target_cpu, .distribution = afft_Distribution_mpst};
}

static inline _afft_ArchParam _afft_makeArchParamMpstGpu(afft_mpst_gpu_Parameters param)
{
  return (_afft_ArchParam){.mpstGpu = param, .target = afft_Target_gpu, .distribution = afft_Distribution_mpst};
}

#define afft_makePlan(planPtr, transformParam, archParam) \
  _afft_makePlan(planPtr, _afft_makeTransformParams(transformParam), _afft_makeArchParam(archParam))

#define AFFT_TYPE_PROPERTIES(_name, _type, _precision_, _complexity) \
  static inline _afft_ExecParam _afft_makeExecParam##_name(_type* ptr) \
  { \
    return (_afft_ExecParam){.ptr = ptr, .precision = _precision_, .complexity = _complexity, .isConst = false}; \
  } \
  \
  static inline _afft_ExecParam _afft_makeExecParamConst##_name(const _type* ptr) \
  { \
    return (_afft_ExecParam){.ptr = (void*)ptr, .precision = _precision_, .complexity = _complexity, .isConst = true}; \
  }

AFFT_TYPE_PROPERTIES(Void, void, afft_Precision_unknown, afft_Complexity_unknown)
AFFT_TYPE_PROPERTIES(Float, float, afft_Precision_f32, afft_Complexity_real)
AFFT_TYPE_PROPERTIES(Double, double, afft_Precision_f64, afft_Complexity_real)
AFFT_TYPE_PROPERTIES(LongDouble, long double, afft_Precision_longDouble, afft_Complexity_real)
#ifndef __STDC_NO_COMPLEX__
  AFFT_TYPE_PROPERTIES(ComplexFloat, float _Complex, afft_Precision_f32, afft_Complexity_complex)
  AFFT_TYPE_PROPERTIES(ComplexDouble, double _Complex, afft_Precision_f64, afft_Complexity_complex)
  AFFT_TYPE_PROPERTIES(ComplexLongDouble, long double _Complex, afft_Precision_longDouble, afft_Complexity_complex)
#endif

#ifndef __cplusplus
# define _afft_makeExecParam(T) _Generic((T), \
    void*:                       _afft_makeExecParamVoid, \
    const void*:                 _afft_makeExecParamConstVoid, \
    float*:                      _afft_makeExecParamFloat, \
    const float*:                _afft_makeExecParamConstFloat, \
    double*:                     _afft_makeExecParamDouble, \
    const double*:               _afft_makeExecParamConstDouble, \
    long double*:                _afft_makeExecParamLongDouble, \
    const long double*:          _afft_makeExecParamConstLongDouble, \
    float _Complex*:             _afft_makeExecParamComplexFloat, \
    const float _Complex*:       _afft_makeExecParamConstComplexFloat, \
    double _Complex*:            _afft_makeExecParamComplexDouble, \
    const double _Complex*:      _afft_makeExecParamConstComplexDouble, \
    long double _Complex*:       _afft_makeExecParamComplexLongDouble, \
    const long double _Complex*: _afft_makeExecParamConstComplexLongDouble)(T)
#else
static inline _afft_ExecParam _afft_makeExecParam(void* ptr)
{
  return _afft_makeExecParamVoid(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const void* ptr)
{
  return _afft_makeExecParamConstVoid(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(float* ptr)
{
  return _afft_makeExecParamFloat(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const float* ptr)
{
  return _afft_makeExecParamConstFloat(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(double* ptr)
{
  return _afft_makeExecParamDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const double* ptr)
{
  return _afft_makeExecParamConstDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(long double* ptr)
{
  return _afft_makeExecParamLongDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const long double* ptr)
{
  return _afft_makeExecParamConstLongDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(float _Complex* ptr)
{
  return _afft_makeExecParamComplexFloat(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const float _Complex* ptr)
{
  return _afft_makeExecParamConstComplexFloat(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(double _Complex* ptr)
{
  return _afft_makeExecParamComplexDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const double _Complex* ptr)
{
  return _afft_makeExecParamConstComplexDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(long double _Complex* ptr)
{
  return _afft_makeExecParamComplexLongDouble(ptr);
}

static inline _afft_ExecParam _afft_makeExecParam(const long double _Complex* ptr)
{
  return _afft_makeExecParamConstComplexLongDouble(ptr);
}
#endif

afft_Error _afft_Plan_execute(afft_Plan* plan, _afft_ExecParam src, _afft_ExecParam dst);

#define afft_Plan_execute(plan, src, dst) \
  _afft_Plan_execute(plan, _afft_makeExecParam(src), _afft_makeExecParam(dst))

afft_Error _afft_Plan_executeUnsafeImpl(afft_Plan* plan, _afft_ExecParam src, _afft_ExecParam dst);

#define afft_Plan_executeUnsafe(plan, src, dst) \
  _afft_Plan_executeUnsafeImpl(plan, _afft_makeExecParam(src), _afft_makeExecParam(dst))

/**********************************************************************************************************************/
// PlanCache
/**********************************************************************************************************************/


#ifdef __cplusplus
}
#endif

#endif /* AFFT_H */
