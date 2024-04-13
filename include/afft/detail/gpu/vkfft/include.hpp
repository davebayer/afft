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

#ifndef AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP
#define AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP

#include "../../../gpu.hpp"

// check if AFFT has been included before including vkFFT
#ifdef VKFFT_H
# error "AFFT and vkFFT cannot be included together in the same translation unit"
#endif

// push the current value of VKFFT_BACKEND
#pragma push_macro("VKFFT_BACKEND")
#undef VKFFT_BACKEND

// push the current value of VKFFT_MAX_FFT_DIMENSIONS
#pragma push_macro("VKFFT_MAX_FFT_DIMENSIONS")
#undef VKFFT_MAX_FFT_DIMENSIONS

// define VKFFT_BACKEND based on the current GPU backend
#if AFFT_GPU_BACKEND_IS_CUDA
# define VKFFT_BACKEND 1
#elif AFFT_GPU_BACKEND_IS_HIP
# define VKFFT_BACKEND 2
#elif AFFT_GPU_BACKEND_IS_OPENCL
# define VKFFT_BACKEND 3
#else
# error "vkFFT backend is only supported with CUDA or HIP"
#endif

// define VKFFT_MAX_FFT_DIMENSIONS based on the maximum number of dimensions
#define VKFFT_MAX_FFT_DIMENSIONS AFFT_MAX_DIM_COUNT

// define the path to the CUDA toolkit if it is not already defined
#if AFFT_GPU_BACKEND_IS_CUDA
# ifndef CUDA_TOOLKIT_ROOT_DIR
#   define CUDA_TOOLKIT_ROOT_DIR AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR
# endif
#endif

// include the vkFFT header
#include <vkFFT.h>

// restore the original value of VKFFT_BACKEND
#pragma pop_macro("VKFFT_BACKEND")

// restore the original value of VKFFT_MAX_FFT_DIMENSIONS
#pragma pop_macro("VKFFT_MAX_FFT_DIMENSIONS")

#endif /* AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP */
