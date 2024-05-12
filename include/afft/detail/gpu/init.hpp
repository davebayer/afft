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

#if AFFT_GPU_BACKEND_IS_ENABLED(CUFFT)
# include "cufft/init.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(HIPFFT)
# include "hipfft/init.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(ROCFFT)
# include "rocfft/init.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(VKFFT)
# include "vkfft/init.hpp"
#endif

namespace afft::detail::gpu
{
  /// @brief Initialize the GPU framework and backends.
  inline void init([[maybe_unused]] const afft::gpu::InitParameters& gpuInitParams)
  {
    // Initialize the GPU framework
# if AFFT_GPU_FRAMEWORK_IS(CUDA)
    cuda::init();
# elif AFFT_GPU_FRAMEWORK_IS(HIP)
    hip::init();
# elif AFFT_GPU_FRAMEWORK_IS(OPENCL)
    opencl::init();
# endif
    
    // Initialize the backends
# if AFFT_GPU_BACKEND_IS_ENABLED(CUFFT)
    cufft::init(gpuInitParams.cufft);
# elif AFFT_GPU_BACKEND_IS_ENABLED(HIPFFT)
    hipfft::init(gpuInitParams.hipfft);
# elif AFFT_GPU_BACKEND_IS_ENABLED(ROCFFT)
    rocfft::init(gpuInitParams.rocfft);
# elif AFFT_GPU_BACKEND_IS_ENABLED(VKFFT)
    vkfft::init(gpuInitParams.vkfft);
# endif
  }

  /// @brief Finalize the GPU framework and backends.
  inline void finalize()
  {
    // Finalize the backends first
# if AFFT_GPU_BACKEND_IS_ENABLED(CUFFT)
    cufft::finalize();
# elif AFFT_GPU_BACKEND_IS_ENABLED(HIPFFT)
    hipfft::finalize();
# elif AFFT_GPU_BACKEND_IS_ENABLED(ROCFFT)
    rocfft::finalize();
# elif AFFT_GPU_BACKEND_IS_ENABLED(VKFFT)
    vkfft::finalize();
# endif

    // Finalize the GPU framework
# if AFFT_GPU_FRAMEWORK_IS(CUDA)
    cuda::finalize();
# elif AFFT_GPU_FRAMEWORK_IS(HIP)
    hip::finalize();
# elif AFFT_GPU_FRAMEWORK_IS(OPENCL)
    opencl::finalize();
# endif
  }
} // namespace afft::detail::gpu

#endif /* AFFT_DETAIL_GPU_INIT_HPP */
