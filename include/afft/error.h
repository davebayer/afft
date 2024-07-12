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

#ifndef AFFT_ERROR_H
#define AFFT_ERROR_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Error enumeration
typedef enum
{
  afft_Error_success,         ///< No error
  afft_Error_internal,        ///< Internal error
  afft_Error_invalidArgument, ///< Invalid argument
  afft_Error_invalidPlan,     ///< Invalid plan
  afft_Error_cudaDriver,      ///< CUDA driver error
  afft_Error_cudaRuntime,     ///< CUDA runtime error
  afft_Error_hip,             ///< HIP error
  afft_Error_opencl,          ///< OpenCL error
  afft_Error_mpi,             ///< MPI error
} afft_Error;

/// @brief Error detail structure
typedef struct
{
  const char* what;          ///< Error message
  union
  {
# ifdef AFFT_ENABLE_CUDA
    CUresult    cudaDriver;  ///< CUDA driver return value, valid only if afft_Error_cudaDriver emitted
    cudaError_t cudaRuntime; ///< CUDA runtime return value, valid only if afft_Error_cudaRuntime emitted
# endif
# ifdef AFFT_ENABLE_HIP
    hipError_t  hip;         ///< HIP return value, valid only if afft_Error_hip emitted
# endif
# ifdef AFFT_ENABLE_OPENCL
    cl_int      opencl;      ///< OpenCL return value, valid only if afft_Error_opencl emitted
# endif
# ifdef AFFT_ENABLE_MPI
    int         mpi;         ///< MPI return value, valid only if afft_Error_mpi emitted
# endif
    int         _dummy;      ///< Dummy value to ensure the union is not empty, do not use
  } retval;
} afft_ErrorDetail;

#ifdef __cplusplus
}
#endif

#endif /* AFFT_ERROR_H */
