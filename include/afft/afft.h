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

#include "detail/include.h"

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

/// @brief Alignment values
enum
{
  afft_Alignment_simd128  = 16,  ///< 128-bit SIMD alignment
  afft_Alignment_simd256  = 32,  ///< 256-bit SIMD alignment
  afft_Alignment_simd512  = 64,  ///< 512-bit SIMD alignment
  afft_Alignment_simd1024 = 128, ///< 1024-bit SIMD alignment
  afft_Alignment_simd2048 = 256, ///< 2048-bit SIMD alignment

  afft_Alignment_sse    = afft_Alignment_simd128,  ///< SSE alignment
  afft_Alignment_sse2   = afft_Alignment_simd128,  ///< SSE2 alignment
  afft_Alignment_sse3   = afft_Alignment_simd128,  ///< SSE3 alignment
  afft_Alignment_sse4   = afft_Alignment_simd128,  ///< SSE4 alignment
  afft_Alignment_sse4_1 = afft_Alignment_simd128,  ///< SSE4.1 alignment
  afft_Alignment_sse4_2 = afft_Alignment_simd128,  ///< SSE4.2 alignment
  afft_Alignment_avx    = afft_Alignment_simd256,  ///< AVX alignment
  afft_Alignment_avx2   = afft_Alignment_simd256,  ///< AVX2 alignment
  afft_Alignment_avx512 = afft_Alignment_simd512,  ///< AVX-512 alignment
  afft_Alignment_neon   = afft_Alignment_simd128,  ///< NEON alignment
  afft_Alignment_sve    = afft_Alignment_simd2048, ///< SVE alignment
};

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
  afft_fftw3_PlannerFlag_exhaustive,      ///< Exhaustive planner flag
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

typedef afft_spst_gpu_clfft_Parameters afft_gpu_clfft_Parameters;
typedef afft_spst_gpu_cufft_Parameters afft_gpu_cufft_Parameters;
typedef afft_spst_cpu_fftw3_Parameters afft_cpu_fftw3_Parameters;

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
    afft_spst_cpu_BackendParameters spstCpu;
    afft_spst_gpu_BackendParameters spstGpu;
    afft_spmt_gpu_BackendParameters spmtGpu;
    afft_mpst_cpu_BackendParameters mpstCpu;
    afft_mpst_gpu_BackendParameters mpstGpu;
  };
  afft_Target                target;
  afft_Distribution          distribution;
} afft_BackendParameters;

static inline afft_BackendParameters _afft_makeBackendParametersSpstCpu(afft_spst_cpu_BackendParameters params)
{
  return (afft_BackendParameters){.spstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_spst};
}

static inline afft_BackendParameters _afft_makeBackendParametersSpstGpu(afft_spst_gpu_BackendParameters params)
{
  return (afft_BackendParameters){.spstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spst};
}

static inline afft_BackendParameters _afft_makeBackendParametersSpmtGpu(afft_spmt_gpu_BackendParameters params)
{
  return (afft_BackendParameters){.spmtGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spmt};
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstCpu(afft_mpst_cpu_BackendParameters params)
{
  return (afft_BackendParameters){.mpstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_mpst};
}

static inline afft_BackendParameters _afft_makeBackendParametersMpstGpu(afft_mpst_gpu_BackendParameters params)
{
  return (afft_BackendParameters){.mpstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_mpst};
}

static inline afft_BackendParameters _afft_makeBackendParametersAny(afft_BackendParameters params)
{
  return params;
}

#ifndef __cplusplus
# define _afft_makeBackendParameters(params) _Generic((params), \
    afft_spst_cpu_BackendParameters: _afft_makeBackendParametersSpstCpu, \
    afft_spst_gpu_BackendParameters: _afft_makeBackendParametersSpstGpu, \
    afft_spmt_gpu_BackendParameters: _afft_makeBackendParametersSpmtGpu, \
    afft_mpst_cpu_BackendParameters: _afft_makeBackendParametersMpstCpu, \
    afft_mpst_gpu_BackendParameters: _afft_makeBackendParametersMpstGpu, \
    afft_BackendParameters:   _afft_makeBackendParametersAny)(params)
#else
static inline _afft_makeBackendParameters(afft_spst_cpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpstCpu(params);
}

static inline _afft_makeBackendParameters(afft_spst_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpstGpu(params);
}

static inline _afft_makeBackendParameters(afft_spmt_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersSpmtGpu(params);
}

static inline _afft_makeBackendParameters(afft_mpst_cpu_BackendParameters params)
{
  return _afft_makeBackendParametersMpstCpu(params);
}

static inline _afft_makeBackendParameters(afft_mpst_gpu_BackendParameters params)
{
  return _afft_makeBackendParametersMpstGpu(params);
}

static inline _afft_makeBackendParameters(afft_BackendParameters params)
{
  return _afft_makeBackendParametersAny(params);
}
#endif

/**
 * @brief Make a plan object.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
#define afft_makePlan(transformParams, archParams, planPtr) \
  _afft_makePlan(_afft_makeTransformParameters(transformParams), \
                 _afft_makeArchitectureParameters(archParams), \
                 planPtr)

/**
 * @brief Make a plan object implementation. Internal use only.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
afft_Error _afft_makePlan(afft_TransformParameters    transformParams,
                          afft_ArchitectureParameters archParams,
                          afft_Plan**                 planPtr);

/**
 * @brief Make a plan object with backend parameters.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param backendParams Backend parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
#define afft_makePlanWithBackendParameters(transformParams, archParams, backendParams, planPtr) \
  _afft_makePlanWithBackendParameters(_afft_makeTransformParameters(transformParams), \
                                      _afft_makeArchitectureParameters(archParams), \
                                      _afft_makeBackendParameters(backendParams), \
                                      planPtr)

/**
 * @brief Make a plan object with backend parameters implementation. Internal use only.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param backendParams Backend parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
afft_Error _afft_makePlanWithBackendParameters(afft_TransformParameters    transformParams,
                                               afft_ArchitectureParameters archParams,
                                               afft_BackendParameters      backendParams,
                                               afft_Plan**                 planPtr);

/**
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTransform(const afft_Plan* plan, afft_Transform* transform);

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTarget(const afft_Plan* plan, afft_Target* target);

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTargetCount(const afft_Plan* plan, size_t* targetCount);

/**
 * @brief Get the plan distribution.
 * @param plan Plan object.
 * @param distribution Pointer to the distribution variable.
 * @return Error code.
 */
afft_Error afft_Plan_getDistribution(const afft_Plan* plan, afft_Distribution* distribution);

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @return Error code.
 */
afft_Error afft_Plan_getBackend(const afft_Plan* plan, afft_Backend* backend);

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param workspaceSize Pointer to the workspace array the same size as number of targets.
 * @return Error code.
 */
afft_Error afft_Plan_getWorkspaceSize(const afft_Plan* plan, size_t* workspaceSize);

typedef struct
{
  union
  {
    void*        ptr;
    void* const* ptrs;
  };
  bool           isSinglePtr;
  bool           isConst;
} _afft_ExecParam;

static inline _afft_ExecParam _afft_makeExecParam_single(void* ptr)
{
  return (_afft_ExecParam){.ptr = ptr, .isSinglePtr = true, .isConst = false};
}

static inline _afft_ExecParam _afft_makeExecParam_singleConst(const void* ptr)
{
  return (_afft_ExecParam){.ptr = (void*)ptr, .isSinglePtr = true, .isConst = true};
}

static inline _afft_ExecParam _afft_makeExecParam_multiple(void* const* ptrs)
{
  return (_afft_ExecParam){.ptrs = ptrs, .isSinglePtr = false, .isConst = false};
}

static inline _afft_ExecParam _afft_makeExecParam_multipleConst(const void* const* ptrs)
{
  return (_afft_ExecParam){.ptrs = (void* const*)ptrs, .isSinglePtr = false, .isConst = true};
}

#ifndef __cplusplus
# define _afft_makeExecParam(ptr) _Generic((ptr), \
    void*:       _afft_makeExecParam_single, \
    const void*: _afft_makeExecParam_singleConst, \
    void* const*: _afft_makeExecParam_multiple, \
    const void* const*: _afft_makeExecParam_multipleConst)(ptr)
#else
static inline _afft_makeExecParam(void* ptr)
{
  return _afft_makeExecParam_single(ptr);
}

static inline _afft_makeExecParam(const void* ptr)
{
  return _afft_makeExecParam_singleConst(ptr);
}

static inline _afft_makeExecParam(void* const* ptrs)
{
  return _afft_makeExecParam_multiple(ptrs);
}

static inline _afft_makeExecParam(const void* const* ptrs)
{
  return _afft_makeExecParam_multipleConst(ptrs);
}
#endif

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @return Error code.
 */
#define afft_Plan_execute(plan, src, dst) \
  _afft_Plan_execute(plan, _afft_makeExecParam(src), _afft_makeExecParam(dst))

/**
 * @brief Execute a plan implementation. Internal use only.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @return Error code.
 */
afft_Error _afft_Plan_execute(afft_Plan* plan, _afft_ExecParam src, _afft_ExecParam dst);

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

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpstCpu(afft_spst_cpu_ExecutionParameters params)
{
  return (afft_ExecutionParameters){.spstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_spst};
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpstGpu(afft_spst_gpu_ExecutionParameters params)
{
  return (afft_ExecutionParameters){.spstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spst};
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersSpmtGpu(afft_spmt_gpu_ExecutionParameters params)
{
  return (afft_ExecutionParameters){.spmtGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_spmt};
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersMpstCpu(afft_mpst_cpu_ExecutionParameters params)
{
  return (afft_ExecutionParameters){.mpstCpu = params, .target = afft_Target_cpu, .distribution = afft_Distribution_mpst};
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersMpstGpu(afft_mpst_gpu_ExecutionParameters params)
{
  return (afft_ExecutionParameters){.mpstGpu = params, .target = afft_Target_gpu, .distribution = afft_Distribution_mpst};
}

static inline afft_ExecutionParameters _afft_makeExecutionParametersAny(afft_ExecutionParameters params)
{
  return params;
}

#ifndef __cplusplus
# define _afft_makeExecutionParameters(params) _Generic((params), \
    afft_spst_cpu_ExecutionParameters: _afft_makeExecutionParametersSpstCpu, \
    afft_spst_gpu_ExecutionParameters: _afft_makeExecutionParametersSpstGpu, \
    afft_spmt_gpu_ExecutionParameters: _afft_makeExecutionParametersSpmtGpu, \
    afft_mpst_cpu_ExecutionParameters: _afft_makeExecutionParametersMpstCpu, \
    afft_mpst_gpu_ExecutionParameters: _afft_makeExecutionParametersMpstGpu, \
    afft_ExecutionParameters:          _afft_makeExecutionParametersAny)(params)
#else
static inline _afft_makeExecutionParameters(afft_spst_cpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpstCpu(params);
}

static inline _afft_makeExecutionParameters(afft_spst_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpstGpu(params);
}

static inline _afft_makeExecutionParameters(afft_spmt_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersSpmtGpu(params);
}

static inline _afft_makeExecutionParameters(afft_mpst_cpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersMpstCpu(params);
}

static inline _afft_makeExecutionParameters(afft_mpst_gpu_ExecutionParameters params)
{
  return _afft_makeExecutionParametersMpstGpu(params);
}

static inline _afft_makeExecutionParameters(afft_ExecutionParameters params)
{
  return _afft_makeExecutionParametersAny(params);
}
#endif

/**
 * @brief Execute a plan with execution parameters.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @param execParams Execution parameters.
 * @return Error code.
 */
#define afft_Plan_executeWithParameters(plan, src, dst, execParams) \
  _afft_Plan_executeWithParameters(plan, \
                                   _afft_makeExecParam(src), \
                                   _afft_makeExecParam(dst), \
                                   _afft_makeExecutionParameters(execParams))

/**
 * @brief Execute a plan with execution parameters implementation. Internal use only.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @param execParams Execution parameters.
 * @return Error code.
 */
afft_Error _afft_Plan_executeWithParameters(const afft_Plan*         plan,
                                            _afft_ExecParam          src,
                                            _afft_ExecParam          dst,
                                            afft_ExecutionParameters execParams);

/**
 * @brief Destroy a plan object.
 * @param plan Plan object.
 */
void afft_Plan_destroy(afft_Plan* plan);

/**********************************************************************************************************************/
// PlanCache
/**********************************************************************************************************************/


#ifdef __cplusplus
}
#endif

#endif /* AFFT_H */
