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

#ifndef AFFT_DETAIL_GPU_INIT_HPP
#define AFFT_DETAIL_GPU_INIT_HPP

#include "../../gpu.hpp"

#if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(CUFFT)
# include "cufft/init.hpp"
#endif
#if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(HIPFFT)
# include "hipfft/init.hpp"
#endif
#if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(ROCFFT)
# include "rocfft/init.hpp"
#endif
#if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(VKFFT)
# include "vkfft/init.hpp"
#endif

namespace afft::detail::gpu
{
  /// @brief Initialize the GPU backend and the transform backend.
  inline void init()
  {
    // Initialize the GPU backend
# if AFFT_GPU_BACKEND_IS_CUDA
    cuda::init();
# elif AFFT_GPU_BACKEND_IS_HIP
    hip::init();
# elif AFFT_GPU_BACKEND_IS_OPENCL
    opencl::init();
# endif
    
    // Initialize the transform backend
# if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(CUFFT)
    cufft::init();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(HIPFFT)
    hipfft::init();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(ROCFFT)
    rocfft::init();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(VKFFT)
    vkfft::init();
# endif
  }

  /// @brief Finalize the GPU backend and the transform backend.
  inline void finalize()
  {
    // Finalize the transform backend first
# if AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(CUFFT)
    cufft::finalize();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(HIPFFT)
    hipfft::finalize();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(ROCFFT)
    rocfft::finalize();
# elif AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(VKFFT)
    vkfft::finalize();
# endif

    // Finalize the GPU backend
# if AFFT_GPU_BACKEND_IS_CUDA
    cuda::finalize();
# elif AFFT_GPU_BACKEND_IS_HIP
    hip::finalize();
# elif AFFT_GPU_BACKEND_IS_OPENCL
    opencl::finalize();
# endif
  }
} // namespace afft::detail::gpu

#endif /* AFFT_DETAIL_GPU_INIT_HPP */
