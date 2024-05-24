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

#ifndef AFFT_DETAIL_INCLUDE_HPP
#define AFFT_DETAIL_INCLUDE_HPP

#include "../config.hpp"

// Include standard C++ headers
#ifndef AFFT_INCLUDE_NO_STD
# if defined(AFFT_CXX_HAS_IMPORT_STD)
import std;
# else
#   include <algorithm>
#   include <array>
#   include <bitset>
#   include <chrono>
#   include <cinttypes>
#   include <climits>
#   include <complex>
#   include <cstddef>
#   include <cstdint>
#   include <cstdio>
#   include <functional>
#   include <limits>
#   include <list>
#   include <memory>
#   include <new>
#   include <optional>
#   if defined(AFFT_CXX_HAS_VERSION) && defined(__cpp_lib_span) && (__cpp_lib_span >= 202002L)
#     include <span>
#   else
#     define TCB_SPAN_NAMESPACE_NAME afft::thirdparty::span
#     include <tcb/span.hpp>
#   endif
#   include <stdexcept>
#   ifdef AFFT_CXX_HAS_STD_FLOAT
#     include <stdfloat>
#   endif
#   include <string>
#   include <string_view>
#   include <tuple>
#   include <type_traits>
#   include <typeinfo>
#   include <unordered_map>
#   include <utility>
#   include <variant>
#   include <vector>
# endif
#endif

// Include GPU backend headers
#if AFFT_GPU_BACKEND_IS(CUDA)
# include <cuda.h>
# include <cuda_runtime.h>
# include <nvrtc.h>
#elif AFFT_GPU_BACKEND_IS(HIP)
# include <hip/hip_runtime.h>
#elif AFFT_GPU_BACKEND_IS(OPENCL)
# if defined(__APPLE__) || defined(__MACOSX)
#   include <OpenCL/cl.h>
# else
#   include <CL/cl.h>
# endif
#endif

// Include multi-processing backend headers
#if AFFT_MP_BACKEND_IS(MPI)
# include <mpi.h>
#endif

// Include clFFT header
#if AFFT_BACKEND_IS_ENABLED(CLFFT)
# if AFFT_GPU_BACKEND_IS(OPENCL)
#   include <clFFT.h>
# endif
#endif

// Include cuFFT header
#if AFFT_BACKEND_IS_ENABLED(CUFFT)
# if AFFT_GPU_BACKEND_IS(CUDA)
#   ifdef AFFT_CUFFT_HAS_MP
#     include <cufftMp.h>
#   else
#     include <cufftXt.h>
#   endif
# endif
#endif

// Include FFTW3 header
#if AFFT_BACKEND_IS_ENABLED(FFTW3)
# if (defined(AFFT_FFTW3_HAS_FLOAT) || defined(AFFT_FFTW3_HAS_DOUBLE) || defined(AFFT_FFTW3_HAS_LONG) || defined(AFFT_FFTW3_HAS_QUAD))
#   include <fftw3.h>
# endif
# if AFFT_MP_BACKEND_IS(MPI) && \
     (defined(AFFT_FFTW3_HAS_MPI_FLOAT) || defined(AFFT_FFTW3_HAS_MPI_DOUBLE) || defined(AFFT_FFTW3_HAS_MPI_LONG))
#   include <fftw3-mpi.h>
# endif
#endif

// Include hipFFT header
#if AFFT_BACKEND_IS_ENABLED(HIPFFT)
# if AFFT_GPU_BACKEND_IS(HIP)
#   include <hipfft/hipfftXt.h>
# endif
#endif

// Include MKL header
#if AFFT_BACKEND_IS_ENABLED(MKL)
# include <mkl_dfti.h>
#endif

// Include PocketFFT header
#if AFFT_BACKEND_IS_ENABLED(POCKETFFT)
# include <pocketfft_hdronly.h>
#endif

#if AFFT_BACKEND_IS_ENABLED(ROCFFT)
// rocFFT header
# if AFFT_GPU_BACKEND_IS(HIP)
#   include <rocfft/rocfft.h>
# endif
#endif

// Include vkFFT header
#if AFFT_BACKEND_IS_ENABLED(VKFFT)
# if AFFT_GPU_BACKEND_IS(CUDA) || \
     (AFFT_GPU_BACKEND_IS(HIP) && defined(__HIP_PLATFORM_AMD__)) || \
     AFFT_GPU_BACKEND_IS(OPENCL)
    // check if AFFT has been included before including vkFFT
#   ifdef VKFFT_H
#     error "AFFT and vkFFT cannot be included together in the same translation unit"
#   endif
    // push the current value of VKFFT_BACKEND
#   pragma push_macro("VKFFT_BACKEND")
#   undef VKFFT_BACKEND
    // push the current value of VKFFT_MAX_FFT_DIMENSIONS
#   pragma push_macro("VKFFT_MAX_FFT_DIMENSIONS")
#   undef VKFFT_MAX_FFT_DIMENSIONS
    // define VKFFT_BACKEND based on the current GPU backend
#   if AFFT_GPU_BACKEND_IS(CUDA)
#     define VKFFT_BACKEND 1
#     ifndef CUDA_TOOLKIT_ROOT_DIR
#       define CUDA_TOOLKIT_ROOT_DIR AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR
#     endif
#   elif AFFT_GPU_BACKEND_IS(HIP)
#     define VKFFT_BACKEND 2
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
#     define VKFFT_BACKEND 3
#   else
#     error "vkFFT backend is only supported with CUDA, HIP or OpenCL"
#   endif
    // define VKFFT_MAX_FFT_DIMENSIONS based on the maximum number of dimensions
#   define VKFFT_MAX_FFT_DIMENSIONS AFFT_MAX_DIM_COUNT
    // include the vkFFT header
#   include <vkFFT.h>
    // restore the original value of VKFFT_BACKEND
#   pragma pop_macro("VKFFT_BACKEND")
    // restore the original value of VKFFT_MAX_FFT_DIMENSIONS
#   pragma pop_macro("VKFFT_MAX_FFT_DIMENSIONS")
# endif
#endif

#endif /* AFFT_DETAIL_INCLUDE_HPP */
