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

/// @brief Macro for CUDA GPU framework
#define AFFT_GPU_FRAMEWORK_CUDA      (1)
/// @brief Macro for HIP GPU framework
#define AFFT_GPU_FRAMEWORK_HIP       (2)
/// @brief Macro for OpenCL GPU framework
#define AFFT_GPU_FRAMEWORK_OPENCL    (3)

/// @brief Macro for checking if GPU is enabled
#define AFFT_GPU_IS_ENABLED          (AFFT_GPU_FRAMEWORK != 0)
/// @brief Macro for checking if CUDA GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_CUDA   (AFFT_GPU_FRAMEWORK == AFFT_GPU_FRAMEWORK_CUDA)
/// @brief Macro for checking if HIP GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_HIP    (AFFT_GPU_FRAMEWORK == AFFT_GPU_FRAMEWORK_HIP)
/// @brief Macro for checking if OpenCL GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_OPENCL (AFFT_GPU_FRAMEWORK == AFFT_GPU_FRAMEWORK_OPENCL)

// Check if GPU framework is defined
#ifndef AFFT_GPU_FRAMEWORK
  // Set GPU framework to 0
# define AFFT_GPU_FRAMEWORK          (0)
#else
  // Check if GPU framework is valid
# if AFFT_GPU_IS_ENABLED && !(AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP || AFFT_GPU_FRAMEWORK_IS_OPENCL)
#   error "Unsupported GPU framework"
# endif
#endif

/// @brief Macro for cufft backend
#define AFFT_GPU_BACKEND_CUFFT  (1 << 0)
/// @brief Macro for hipfft backend
#define AFFT_GPU_BACKEND_HIPFFT (1 << 1)
/// @brief Macro for rocfft backend
#define AFFT_GPU_BACKEND_ROCFFT (1 << 2)
/// @brief Macro for vkfft backend
#define AFFT_GPU_BACKEND_VKFFT  (1 << 3)

/**
 * @brief Implementation of AFFT_GPU_BACKEND_FROM_NAME
 * @param backendName Name of the backend
 * @return Backend id
 * @warning Do not use this macro directly
 */
#define AFFT_GPU_BACKEND_FROM_NAME_IMPL(backendName) \
  AFFT_GPU_BACKEND_##backendName

/**
 * @brief Macro for getting the backend from the name
 * @param backendName Name of the backend
 * @return Backend id
 */
#define AFFT_GPU_BACKEND_FROM_NAME(backendName) \
  AFFT_GPU_BACKEND_FROM_NAME_IMPL(backendName)

/**
 * @brief Implementation of AFFT_GPU_BACKEND_MASK
 * @param ... List of backend names
 * @return Backend mask
 * @warning Do not use this macro directly
 */
#define AFFT_GPU_BACKEND_MASK_IMPL(...) \
  AFFT_BITOR(AFFT_FOR_EACH_WITH_DELIM(AFFT_GPU_BACKEND_FROM_NAME, AFFT_DELIM_COMMA __VA_OPT__(,) __VA_ARGS__))

/**
 * @brief Macro for getting the backend mask
 * @return Backend mask
 * @warning Requires AFFT_GPU_BACKEND_LIST to be defined
 */
#define AFFT_GPU_BACKEND_MASK \
  (AFFT_GPU_BACKEND_ALLOWED_MASK & AFFT_GPU_BACKEND_MASK_IMPL(AFFT_GPU_BACKEND_LIST))

/**
 * @brief Macro for checking if the backend is enabled
 * @param backendName Name of the backend
 * @return Non zero if the backend is enabled, zero otherwise
 */
#define AFFT_GPU_BACKEND_IS_ENABLED(backendName) \
  (AFFT_GPU_BACKEND_FROM_NAME(backendName) & AFFT_DETAIL_EXPAND(AFFT_GPU_BACKEND_MASK))

// Set the mask of enabled backends for given GPU framework
#if AFFT_GPU_FRAMEWORK_IS_CUDA
# define AFFT_GPU_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_BACKEND_CUFFT | AFFT_GPU_BACKEND_VKFFT)
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# define AFFT_GPU_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_BACKEND_HIP | AFFT_GPU_BACKEND_ROCFFT | AFFT_GPU_BACKEND_VKFFT)
#elif AFFT_GPU_FRAMEWORK_IS_OPENCL
# define AFFT_GPU_BACKEND_ALLOWED_MASK 0
#else
# define AFFT_GPU_BACKEND_ALLOWED_MASK 0
#endif

// Include the appropriate GPU framework
#if AFFT_GPU_FRAMEWORK_IS_CUDA
# include "detail/gpu/cuda/cuda.hpp"
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# include "detail/gpu/hip/hip.hpp"
#elif AFFT_GPU_FRAMEWORK_IS_OPENCL
# include "detail/gpu/opencl/opencl.hpp"
#endif

#include <array>
#include <span>

namespace afft::gpu
{
  /// @brief Enum class for backends
  enum class Backend
  {
    cufft,
    hipfft,
    rocfft,
    vkfft,
  };
  
  /// @brief Number of backends
  inline constexpr std::size_t backendCount{4};

  namespace cufft
  {
    /// @brief Init parameters for cufft backend
    struct InitParameters {};
  } // namespace cufft

  namespace hipfft
  {
    /// @brief Init parameters for hipfft backend
    struct InitParameters {};
  } // namespace hipfft

  namespace rocfft
  {
    /// @brief Init parameters for rocfft backend
    struct InitParameters {};
  } // namespace rocfft

  namespace vkfft
  {
    /// @brief Init parameters for vkfft backend
    struct InitParameters {};
  } // namespace vkfft

  /**
   * @struct InitParameters
   * @brief Init parameters for GPU backend
   */
  struct InitParameters
  {
    cufft::InitParameters  cufft{};  ///< Parameters for cufft backend
    hipfft::InitParameters hipfft{}; ///< Parameters for hipfft backend
    rocfft::InitParameters rocfft{}; ///< Parameters for rocfft backend
    vkfft::InitParameters  vkfft{};  ///< Parameters for vkfft backend
  };

  /**
   * @struct Parameters
   * @brief Parameters for GPU backend
   */
  struct Parameters
  {
  // GPU framework specific parameters
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    int  device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    int  device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
# endif
    bool externalWorkspace{false};                      ///< Use external workspace, defaults to `false`
  };

  /// @brief Default list of backends
  inline constexpr std::array defaultBackendList
  {
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    Backend::cufft,  // prefer cufft
    Backend::vkfft,  // fallback to vkfft
# elif AFFT_GPU_FRAMEWORK_IS_HIP && defined(__HIP_PLATFORM_NVIDIA__)
    Backend::hipfft, // prefer cufft (it is used by hipfft on CUDA)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
# elif AFFT_GPU_FRAMEWORK_IS_HIP && defined(__HIP_PLATFORM_AMD__)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
# else
    Backend::vkfft,  // fallback to vkfft as it supports all platforms
# endif
  };

  /// @brief Select parameters for backends
  struct BackendSelectParameters
  {
    std::span<const Backend> backends{defaultBackendList};           ///< Priority list of allowed backends
    BackendSelectStrategy    strategy{BackendSelectStrategy::first}; ///< Backend select strategy
  };

  /**
   * @struct ExecutionParameters
   * @brief Execution parameters for GPU backend
   */
  struct ExecutionParameters
  {
  // GPU framework specific execution parameters
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    cudaStream_t stream{0};   ///< CUDA stream, defaults to `zero` stream
    void*        workspace{}; ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    hipStream_t stream{0};    ///< HIP stream, defaults to `zero` stream
    void*       workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
// # elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    // cl_command_queue commandQueue{};
    // cl_mem           workspace{};
# endif
  };
} // namespace afft

#endif /* AFFT_GPU_HPP */
