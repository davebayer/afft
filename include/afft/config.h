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

#ifndef AFFT_EXTERNAL_CONFIG
# include "afft-config.h"
#endif

// If max dimension count is not defined, use 4 as default
#ifdef AFFT_MAX_DIM_COUNT
# if AFFT_MAX_DIM_COUNT < 1
#   error "AFFT_MAX_DIM_COUNT must be at least 1"
# endif
#else
# define AFFT_MAX_DIM_COUNT     4
#endif

#if defined(AFFT_ENABLE_CUDA) && !defined(AFFT_ENABLE_HIP) && !defined(AFFT_ENABLE_OPENCL)
# ifndef AFFT_CUDA_ROOT_DIR
#   error "AFFT_CUDA_ROOT_DIR must be defined"
# endif
#elif !defined(AFFT_ENABLE_CUDA) && defined(AFFT_ENABLE_HIP) && !defined(AFFT_ENABLE_OPENCL)
# ifndef AFFT_HIP_ROOT_DIR
#   error "AFFT_HIP_ROOT_DIR must be defined"
# endif
#elif !defined(AFFT_ENABLE_CUDA) && !defined(AFFT_ENABLE_HIP) && defined(AFFT_ENABLE_OPENCL)
#elif !defined(AFFT_ENABLE_CUDA) && !defined(AFFT_ENABLE_HIP) && !defined(AFFT_ENABLE_OPENCL)
# define AFFT_DISABLE_GPU
#else
# error "Exactly one GPU backend must be enabled"
#endif

#if defined(AFFT_ENABLE_MPI)
#endif

/**
 * @brief Define AFFT_PARAM macro to enable passing parameters containing commas
 * @param ... Parameter
 */
#define AFFT_PARAM(...)         __VA_ARGS__

#endif /* AFFT_CONFIG_H */
