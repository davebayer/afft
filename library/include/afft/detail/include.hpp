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

#include "config.hpp"

// Include standard C++ headers
#ifndef AFFT_INCLUDE_NO_STD
# include <algorithm>
# include <array>
# include <bitset>
# include <chrono>
# include <cinttypes>
# include <climits>
# include <charconv>
# include <complex>
# include <cstddef>
# include <cstdint>
# include <cstdio>
# include <cstring>
# ifdef AFFT_CXX_HAS_FORMAT
#   include <format>
# endif
# include <functional>
# include <iterator>
# include <limits>
# include <list>
# include <memory>
# include <new>
# include <numeric>
# include <optional>
# ifdef AFFT_CXX_HAS_SPAN
#   include <span>
# else
#   define TCB_SPAN_NAMESPACE_NAME afft::thirdparty::span
#   include <tcb/span.hpp>
# endif
# include <stdexcept>
# ifdef AFFT_CXX_HAS_STD_FLOAT
#   include <stdfloat>
# endif
# include <string>
# include <string_view>
# include <tuple>
# include <type_traits>
# include <typeinfo>
# include <unordered_map>
# include <utility>
# include <variant>
# include <vector>
#endif

// Include afft C API
#include "../afft.h"

#ifdef AFFT_HEADER_ONLY
 // Include clFFT header
# ifdef AFFT_ENABLE_CLFFT
#   include <clFFT.h>
# endif

 // Include cuFFT header
# ifdef AFFT_ENABLE_CUFFT
#   ifdef AFFT_CUFFT_HAS_MP
#     include <cufftMp.h>
#   else
#     include <cufftXt.h>
#   endif
#   if CUFFT_VERSION < 8000
#     error "cuFFT version 8.0 or higher is required"
#   endif
# endif

 // Include FFTW3 header
# ifdef AFFT_ENABLE_FFTW3
#   if (defined(AFFT_FFTW3_HAS_FLOAT) || defined(AFFT_FFTW3_HAS_DOUBLE) || defined(AFFT_FFTW3_HAS_LONG) || defined(AFFT_FFTW3_HAS_QUAD))
#     include <fftw3.h>
#   endif
#   if defined(AFFT_FFTW3_HAS_MPI_FLOAT) || defined(AFFT_FFTW3_HAS_MPI_DOUBLE) || defined(AFFT_FFTW3_HAS_MPI_LONG)
#     include <fftw3-mpi.h>
#   endif
# endif

 // Include HeFFTe header
# ifdef AFFT_ENABLE_HEFFTE
#   include <heffte.h>
# endif

 // Include hipFFT header
# ifdef AFFT_ENABLE_HIPFFT
#   include <hipfft/hipfftXt.h>
#   include <hipfft/hipfft-version.h>
# endif

 // Include MKL header
# ifdef AFFT_ENABLE_MKL
#   include <mkl.h>
#   ifdef AFFT_MKL_HAS_OMP_OFFLOAD
#     include <mkl_omp_offload.h>
#   endif
#   ifdef AFFT_MKL_HAS_CDFT
#     include <mkl_cdft.h>
#   endif
# endif

 // Include PocketFFT header
# ifdef AFFT_ENABLE_POCKETFFT
#   include <pocketfft_hdronly.h>
# endif

 // Include rocFFT header
# ifdef AFFT_ENABLE_ROCFFT
#   include <rocfft/rocfft.h>
#   include <rocfft/rocfft-version.h>
# endif

 // Include vkFFT header
# ifdef AFFT_ENABLE_VKFFT
#   if defined(AFFT_ENABLE_CUDA) || \
       (defined(AFFT_ENABLE_HIP) && defined(__HIP_PLATFORM_AMD__)) || \
       defined(AFFT_ENABLE_OPENCL)
      // check if AFFT has been included before including vkFFT
#     ifdef VKFFT_H
#       error "AFFT and vkFFT cannot be included together in the same translation unit"
#     endif
      // push the current value of VKFFT_BACKEND
#     pragma push_macro("VKFFT_BACKEND")
#     undef VKFFT_BACKEND
      // push the current value of VKFFT_MAX_FFT_DIMENSIONS
#     pragma push_macro("VKFFT_MAX_FFT_DIMENSIONS")
#     undef VKFFT_MAX_FFT_DIMENSIONS
      // push the current value of VKFFT_USE_DOUBLEDOUBLE_FP128
#     pragma push_macro("VKFFT_USE_DOUBLEDOUBLE_FP128")
#     undef VKFFT_USE_DOUBLEDOUBLE_FP128

      // define VKFFT_BACKEND based on the selected backend
#     define VKFFT_BACKEND AFFT_VKFFT_BACKEND
      // define CUDA_TOOLKIT_ROOT_DIR if using CUDA backend
#     if AFFT_VKFFT_BACKEND == 1
#       ifndef CUDA_TOOLKIT_ROOT_DIR
#         define CUDA_TOOLKIT_ROOT_DIR AFFT_CUDA_ROOT_DIR
#       endif
#     endif
      // define VKFFT_MAX_FFT_DIMENSIONS based on the maximum number of dimensions
#     define VKFFT_MAX_FFT_DIMENSIONS (AFFT_MAX_DIM_COUNT + 1)
      // define VKFFT_USE_DOUBLEDOUBLE_FP128 if double-double precision is enabled
#     ifdef AFFT_VKFFT_HAS_DOUBLE_DOUBLE
#       define VKFFT_USE_DOUBLEDOUBLE_FP128
#     endif
      // include the vkFFT header
#     include <vkFFT.h>
      // restore the original value of VKFFT_BACKEND
#     pragma pop_macro("VKFFT_BACKEND")
      // restore the original value of VKFFT_MAX_FFT_DIMENSIONS
#     pragma pop_macro("VKFFT_MAX_FFT_DIMENSIONS")
      // restore the original value of VKFFT_USE_DOUBLEDOUBLE_FP128
#     pragma pop_macro("VKFFT_USE_DOUBLEDOUBLE_FP128")
#   endif
# endif
#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_INCLUDE_HPP */
