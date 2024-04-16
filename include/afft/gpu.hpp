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

#ifndef AFFT_GPU_HPP
#define AFFT_GPU_HPP

#include "macro.hpp"

/// @brief Macro for checking if GPU backend is enabled
#define AFFT_GPU_ENABLED           (AFFT_GPU_BACKEND != 0)

/// @brief Macro for CUDA GPU backend
#define AFFT_GPU_BACKEND_CUDA      (1)
/// @brief Macro for HIP GPU backend
#define AFFT_GPU_BACKEND_HIP       (2)
/// @brief Macro for OpenCL GPU backend
#define AFFT_GPU_BACKEND_OPENCL    (3)

/// @brief Macro for checking if CUDA GPU backend is selected
#define AFFT_GPU_BACKEND_IS_CUDA   (AFFT_GPU_BACKEND == AFFT_GPU_BACKEND_CUDA)
/// @brief Macro for checking if HIP GPU backend is selected
#define AFFT_GPU_BACKEND_IS_HIP    (AFFT_GPU_BACKEND == AFFT_GPU_BACKEND_HIP)
/// @brief Macro for checking if OpenCL GPU backend is selected
#define AFFT_GPU_BACKEND_IS_OPENCL (AFFT_GPU_BACKEND == AFFT_GPU_BACKEND_OPENCL)

// Check if GPU backend is defined
#ifndef AFFT_GPU_BACKEND
  // Set GPU backend to 0
# define AFFT_GPU_BACKEND          (0)
#else
  // Check if GPU backend is valid
# if AFFT_GPU_ENABLED && !(AFFT_GPU_BACKEND_IS_CUDA || AFFT_GPU_BACKEND_IS_HIP || AFFT_GPU_BACKEND_IS_OPENCL)
#   error "Unsupported GPU backend"
# endif
#endif

/// @brief Macro for cufft transform backend
#define AFFT_GPU_TRANSFORM_BACKEND_CUFFT  (1 << 0)
/// @brief Macro for hipfft transform backend
#define AFFT_GPU_TRANSFORM_BACKEND_HIPFFT (1 << 1)
/// @brief Macro for rocfft transform backend
#define AFFT_GPU_TRANSFORM_BACKEND_ROCFFT (1 << 2)
/// @brief Macro for vkfft transform backend
#define AFFT_GPU_TRANSFORM_BACKEND_VKFFT  (1 << 3)

/**
 * @brief Implementation of AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME
 * @param backendName Name of the backend
 * @return Transform backend
 * @warning Do not use this macro directly
 */
#define AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName) \
  AFFT_GPU_TRANSFORM_BACKEND_##backendName

/**
 * @brief Macro for getting the transform backend from the name
 * @param backendName Name of the backend
 * @return Transform backend
 */
#define AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME(backendName) \
  AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName)

/**
 * @brief Implementation of AFFT_GPU_TRANSFORM_BACKEND_MASK
 * @param ... List of transform backend names
 * @return Transform backend mask
 * @warning Do not use this macro directly
 */
#define AFFT_GPU_TRANSFORM_BACKEND_MASK_IMPL(...) \
  AFFT_BITOR(AFFT_FOR_EACH_WITH_DELIM(AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME, AFFT_DELIM_COMMA __VA_OPT__(,) __VA_ARGS__))

/**
 * @brief Macro for getting the transform backend mask
 * @return Transform backend mask
 * @warning Requires AFFT_GPU_TRANSFORM_BACKEND_LIST to be defined
 */
#define AFFT_GPU_TRANSFORM_BACKEND_MASK \
  (AFFT_GPU_TRANSFORM_BACKEND_ALLOWED_MASK & AFFT_GPU_TRANSFORM_BACKEND_MASK_IMPL(AFFT_GPU_TRANSFORM_BACKEND_LIST))

/**
 * @brief Macro for checking if the transform backend is allowed
 * @param backendName Name of the backend
 * @return Non zero if the transform backend is allowed, zero otherwise
 */
#define AFFT_GPU_TRANSFORM_BACKEND_IS_ALLOWED(backendName) \
  (AFFT_GPU_TRANSFORM_BACKEND_FROM_NAME(backendName) & AFFT_DETAIL_EXPAND(AFFT_GPU_TRANSFORM_BACKEND_MASK))

// Set the mask of allowed transform backends for given GPU backend
#if AFFT_GPU_BACKEND_IS_CUDA
# define AFFT_GPU_TRANSFORM_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_TRANSFORM_BACKEND_CUFFT | AFFT_GPU_TRANSFORM_BACKEND_VKFFT)
#elif AFFT_GPU_BACKEND_IS_HIP
# define AFFT_GPU_TRANSFORM_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_TRANSFORM_BACKEND_HIP | AFFT_GPU_TRANSFORM_BACKEND_ROCFFT | AFFT_GPU_TRANSFORM_BACKEND_VKFFT)
#elif AFFT_GPU_BACKEND_IS_OPENCL
# define AFFT_GPU_TRANSFORM_BACKEND_ALLOWED_MASK 0
#else
# define AFFT_GPU_TRANSFORM_BACKEND_ALLOWED_MASK 0
#endif

// Include the appropriate GPU backend
#if AFFT_GPU_BACKEND_IS_CUDA
# include "detail/gpu/cuda/cuda.hpp"
#elif AFFT_GPU_BACKEND_IS_HIP
# include "detail/gpu/hip/hip.hpp"
#elif AFFT_GPU_BACKEND_IS_OPENCL
# include "detail/gpu/opencl/opencl.hpp"
#endif

#include <span>

namespace afft::gpu
{
  /// @brief Enum class for transform backend
  enum class TransformBackend
  {
    cufft  = AFFT_GPU_TRANSFORM_BACKEND_CUFFT,
    hipfft = AFFT_GPU_TRANSFORM_BACKEND_HIPFFT,
    rocfft = AFFT_GPU_TRANSFORM_BACKEND_ROCFFT,
    vkfft  = AFFT_GPU_TRANSFORM_BACKEND_VKFFT,
  };

  /**
   * @struct Parameters
   * @brief Parameters for GPU backend
   */
  struct Parameters
  {
  // GPU backend specific parameters
# if AFFT_GPU_BACKEND_IS_CUDA
    int  device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
# elif AFFT_GPU_BACKEND_IS_HIP
    int  device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
# endif
    bool externalWorkspace{false};                      ///< Use external workspace, defaults to `false`
  };

  /**
   * @struct ExecutionParameters
   * @brief Execution parameters for GPU backend
   */
  struct ExecutionParameters
  {
  // GPU backend specific execution parameters
# if AFFT_GPU_BACKEND_IS_CUDA
    cudaStream_t stream{0};   ///< CUDA stream, defaults to `zero` stream
    void*        workspace{}; ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_BACKEND_IS_HIP
    hipStream_t stream{0};    ///< HIP stream, defaults to `zero` stream
    void*       workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
// # elif AFFT_GPU_BACKEND_IS_OPENCL
    // cl_command_queue commandQueue{};
    // cl_mem           workspace{};
# endif
  };
} // namespace afft

#endif /* AFFT_GPU_HPP */
