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

#ifndef AFFT_DETAIL_GPU_HIP_INCLUDE_HPP
#define AFFT_DETAIL_GPU_HIP_INCLUDE_HPP

#if defined(AFFT_GPU_HIP_PLATFORM_AMD) && !defined(AFFT_GPU_HIP_PLATFORM_NVIDIA)
  // Check if the HIP platform detected by the compiler matches the AFFT HIP platform
# if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
#   error "AFFT HIP platform does not match the HIP platform detected by the compiler"
# endif
  // Define the AMD HIP platform
# ifndef __HIP_PLATFORM_AMD__
#   define __HIP_PLATFORM_AMD__
# endif
# ifndef __HIP_PLATFORM_HCC__
#   define __HIP_PLATFORM_HCC__
# endif
#elif defined(AFFT_GPU_HIP_PLATFORM_NVIDIA) && !defined(AFFT_GPU_HIP_PLATFORM_AMD)
  // Check if the HIP platform detected by the compiler matches the AFFT HIP platform
# if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#   error "AFFT HIP platform does not match the HIP platform detected by the compiler"
# endif
  // Define the NVIDIA HIP platform
# ifndef __HIP_PLATFORM_NVIDIA__
#   define __HIP_PLATFORM_NVIDIA__
# endif
# ifndef __HIP_PLATFORM_NVCC__
#   define __HIP_PLATFORM_NVCC__
# endif
#elif defined(AFFT_GPU_HIP_PLATFORM_AMD) && defined(AFFT_GPU_HIP_PLATFORM_NVIDIA)
# error "Only one of AFFT_GPU_HIP_PLATFORM_AMD and AFFT_GPU_HIP_PLATFORM_NVIDIA can be defined"
#else
// AFFT HIP platform is not defined, let the compilation fail on HIP include
#endif

// Include HIP header
#include <hip/hip_runtime.h>

#endif /* AFFT_DETAIL_GPU_HIP_INCLUDE_HPP */
