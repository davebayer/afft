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

#ifndef AFFT_CONFIG_H
#define AFFT_CONFIG_H

#include <afft/afft-version.h>

#ifndef AFFT_EXTERNAL_CONFIG
# include <afft/afft-config.h>
#endif

// If max dimension count is not defined, use 4 as default
#ifdef AFFT_MAX_RANK
# if AFFT_MAX_RANK < 1
#   error "AFFT_MAX_RANK must be at least 1"
# endif
#else
# define AFFT_MAX_RANK 4
#endif

#ifdef AFFT_ENABLE_CUDA
  // Check if AFFT_CUDA_TOOLKIT_ROOT_DIR is defined when CUDA is enabled
# ifndef AFFT_CUDA_TOOLKIT_ROOT_DIR
#   error "AFFT_CUDA_TOOLKIT_ROOT_DIR must be defined"
# endif

  // Check if AFFT_CUDA_MAX_DEVICES is defined when CUDA is enabled
# ifdef AFFT_CUDA_MAX_DEVICES
#   if AFFT_CUDA_MAX_DEVICES < 1
#     error "AFFT_CUDA_MAX_DEVICES must be at least 1"
#   endif
# else
#   define AFFT_CUDA_MAX_DEVICES 16
# endif
#endif

#ifdef AFFT_ENABLE_HIP
  // Check if AFFT_HIP_MAX_DEVICES is defined when HIP is enabled
# ifndef AFFT_HIP_MAX_DEVICES
#   if AFFT_HIP_MAX_DEVICES < 1
#     error "AFFT_HIP_MAX_DEVICES must be at least 1"
#   endif
# else
#   define AFFT_HIP_MAX_DEVICES 16
# endif
#endif

// FFTW3 quad precision is only supported with GCC version 4.6 or higher
#ifdef AFFT_ENABLE_FFTW3
# if !((__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
       && !(defined(__ICC) || defined(__INTEL_COMPILER) || defined(__CUDACC__) || defined(__PGI)) \
       && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__)))
#   ifdef AFFT_FFTW3_HAS_QUAD
#     undef AFFT_FFTW3_HAS_QUAD
#   endif
# endif
#endif

/**
 * @brief Define AFFT_PARAM macro to enable passing parameters containing commas
 * @param[in] ... Parameter
 */
#define AFFT_PARAM(...)         __VA_ARGS__

#endif /* AFFT_CONFIG_H */
