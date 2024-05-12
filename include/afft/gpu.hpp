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

// Check if GPU framework is defined (GPU support is enabled)
#ifdef AFFT_GPU_FRAMEWORK
  /**
   * @brief Macro for getting the GPU framework from the name
   * @param frameworkName Name of the framework
   * @return Framework id
   */
# define AFFT_GPU_FRAMEWORK_FROM_NAME(frameworkName) \
    AFFT_DETAIL_EXPAND_AND_CONCAT(AFFT_GPU_FRAMEWORK_, frameworkName)

  /// @brief Macro for checking if GPU support is enabled
# define AFFT_GPU_IS_ENABLED                       (1)

  /**
   * @brief Implementation of AFFT_GPU_FRAMEWORK_IS_* macros
   * @param frameworkName Name of the framework
   * @return Non zero if the framework is selected, zero otherwise
   */
# define AFFT_GPU_FRAMEWORK_IS_IMPL(frameworkName) \
    (AFFT_GPU_FRAMEWORK_FROM_NAME(frameworkName) == AFFT_GPU_FRAMEWORK_FROM_NAME(AFFT_GPU_FRAMEWORK))
#else
# define AFFT_GPU_IS_ENABLED                       (0)
# define AFFT_GPU_FRAMEWORK_IS_IMPL(frameworkName) (0)
#endif

/// @brief Macro for checking if CUDA GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_CUDA   (AFFT_GPU_FRAMEWORK_IS_IMPL(CUDA))
/// @brief Macro for checking if HIP GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_HIP    (AFFT_GPU_FRAMEWORK_IS_IMPL(HIP))
/// @brief Macro for checking if OpenCL GPU framework is selected
#define AFFT_GPU_FRAMEWORK_IS_OPENCL (AFFT_GPU_FRAMEWORK_IS_IMPL(OPENCL))

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

/// @brief Macro for clfft backend
#define AFFT_GPU_BACKEND_CLFFT  (1 << 0)
/// @brief Macro for cufft backend
#define AFFT_GPU_BACKEND_CUFFT  (1 << 1)
/// @brief Macro for hipfft backend
#define AFFT_GPU_BACKEND_HIPFFT (1 << 2)
/// @brief Macro for rocfft backend
#define AFFT_GPU_BACKEND_ROCFFT (1 << 3)
/// @brief Macro for vkfft backend
#define AFFT_GPU_BACKEND_VKFFT  (1 << 4)

/**
 * @brief Macro for getting the backend from the name
 * @param backendName Name of the backend
 * @return Backend id
 */
#define AFFT_GPU_BACKEND_FROM_NAME(backendName) \
  AFFT_DETAIL_EXPAND_AND_CONCAT(AFFT_GPU_BACKEND_, backendName)

/**
 * @brief Macro for checking if the backend is enabled
 * @param backendName Name of the backend
 * @return Non zero if the backend is enabled, zero otherwise
 */
#define AFFT_GPU_BACKEND_IS_ENABLED(backendName) \
  (AFFT_GPU_BACKEND_FROM_NAME(backendName) & (AFFT_GPU_BACKEND_MASK))

// Set the mask of enabled backends for given GPU framework
#if AFFT_GPU_FRAMEWORK_IS_CUDA
# define AFFT_GPU_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_BACKEND_CUFFT | AFFT_GPU_BACKEND_VKFFT)
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# ifdef __HIP_PLATFORM_AMD__
#   define AFFT_GPU_BACKEND_ALLOWED_MASK \
      (AFFT_GPU_BACKEND_HIPFFT | AFFT_GPU_BACKEND_ROCFFT | AFFT_GPU_BACKEND_VKFFT)
# else
#   define AFFT_GPU_BACKEND_ALLOWED_MASK \
      (AFFT_GPU_BACKEND_HIPFFT | AFFT_GPU_BACKEND_ROCFFT)
# endif
#elif AFFT_GPU_FRAMEWORK_IS_OPENCL
# define AFFT_GPU_BACKEND_ALLOWED_MASK \
    (AFFT_GPU_BACKEND_CLFFT | AFFT_GPU_BACKEND_VKFFT)
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

#include "backend.hpp"
#include "common.hpp"

namespace afft::gpu
{
#if AFFT_GPU_FRAMEWORK_IS_CUDA
  /// @brief Backend mask for CUDA GPU framework
  inline constexpr BackendMask backendMask{Backend::cufft | Backend::vkfft};
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# if defined(__HIP_PLATFORM_AMD__)
  /// @brief Backend mask for HIP GPU framework on AMD platform
  inline constexpr BackendMask backendMask{Backend::vkfft | Backend::rocfft};
# elif defined(__HIP_PLATFORM_NVIDIA__)
  /// @brief Backend mask for HIP GPU framework on NVIDIA platform
  inline constexpr BackendMask backendMask{Backend::hipfft | Backend::vkfft | Backend::rocfft};
# endif
#elif AFFT_GPU_FRAMEWORK_IS_OPENCL
  /// @brief Backend mask for OpenCL GPU framework
  inline constexpr BackendMask backendMask{Backend::clfft | Backend::vkfft};
#endif

  /// @brief Default backend initialization order
  inline constexpr std::array defaultBackendInitOrder
  {
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    Backend::cufft,  // prefer cufft
    Backend::vkfft,  // fallback to vkfft
# elif AFFT_GPU_FRAMEWORK_IS_HIP
#   if defined(__HIP_PLATFORM_AMD__)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
#   elif defined(__HIP_PLATFORM_NVIDIA__)
    Backend::hipfft, // prefer cufft (it is used by hipfft on CUDA)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
#  endif
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    Backend::vkfft,  // prefer vkfft
    Backend::clfft,  // fallback to clfft
# else
    Backend::vkfft,  // fallback to vkfft as it supports all platforms
# endif
  };

  /**
   * @struct Parameters
   * @brief Parameters for GPU backend
   */
  struct Parameters
  {
    MemoryLayout    memoryLayout{};                                ///< Memory layout for CPU transform
    ComplexFormat   complexFormat{ComplexFormat::interleaved};     ///< complex number format
    bool            destroySource{false};                          ///< destroy source data
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< workspace policy
    // GPU framework specific parameters
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    int             device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    int             device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context      context{};                                     ///< OpenCL context
    cl_device_id    device{};                                      ///< OpenCL device
# endif
    bool            externalWorkspace{false};                      ///< Use external workspace, defaults to `false`
  };

  /**
   * @struct ExecutionParameters
   * @brief Execution parameters for gpu target
   */
  struct ExecutionParameters
  {
  // GPU framework specific execution parameters
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    cudaStream_t     stream{0};   ///< CUDA stream, defaults to `zero` stream
    void*            workspace{}; ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    hipStream_t      stream{0};    ///< HIP stream, defaults to `zero` stream
    void*            workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_command_queue commandQueue{};
    cl_mem           workspace{};
# endif
  };

  /**
   * @class UnifiedMemoryAllocator
   * @brief Allocator named concept implementation implementation for unified GPU memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
  class UnifiedMemoryAllocator
  {
    public:
      /// @brief Type of the memory
      using value_type = T;

#   if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
      /// @brief Default constructor
      constexpr UnifiedMemoryAllocator() noexcept = default;
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
      /// @brief Default constructor
      UnifiedMemoryAllocator() = delete;

      /// @brief Constructor
      constexpr UnifiedMemoryAllocator(cl_context context) noexcept
      : mContext(context)
      {}
#   endif

      /// @brief Copy constructor
      template<typename U>
      constexpr UnifiedMemoryAllocator([[maybe_unused]] const UnifiedMemoryAllocator<U>& other) noexcept
#   if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
      : mContext(other.context)
#   endif
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr UnifiedMemoryAllocator([[maybe_unused]] UnifiedMemoryAllocator<U>&& other) noexcept
#   if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
      : mContext(std::move(other.context))
#   endif
      {}

      /// @brief Destructor
      ~UnifiedMemoryAllocator() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr UnifiedMemoryAllocator& operator=(const UnifiedMemoryAllocator<U>& other) noexcept
      {
        if (this != &other)
        {
#       if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
#       elif AFFT_GPU_FRAMEWORK_IS_OPENCL
          mContext = other.context;
#       endif
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr UnifiedMemoryAllocator& operator=(UnifiedMemoryAllocator<U>&& other) noexcept
      {
        if (this != &other)
        {
#       if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
#       elif AFFT_GPU_FRAMEWORK_IS_OPENCL
          mContext = std::move(other.context);
#       endif
        }
        return *this;
      }

      /**
       * @brief Allocate memory
       * @param n Number of elements
       * @return Pointer to the allocated memory
       */
      [[nodiscard]] T* allocate(std::size_t n)
      {
        T* ptr{};

        [[maybe_unused]] const std::size_t sizeInBytes = n * sizeof(T);

#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        detail::Error::check(cudaMallocManaged(&ptr, sizeInBytes));
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        detail::Error::check(hipMallocManaged(&ptr, sizeInBytes));
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        ptr = static_cast<T*>(clSVMAlloc(mContext, CL_MEM_READ_WRITE, sizeInBytes, 0));
#     endif

        if (ptr == nullptr)
        {
          throw std::bad_alloc();
        }

        return ptr;
      }

      /**
       * @brief Deallocate memory
       * @param p Pointer to the memory
       * @param n Number of elements
       */
      void deallocate([[maybe_unused]] T* p, std::size_t) noexcept
      {
#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        detail::Error::check(cudaFree(p));
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        detail::Error::check(hipFree(p));
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        clSVMFree(mContext, p);
#     endif
      }

#   if AFFT_GPU_FRAMEWORK_IS_OPENCL
      /// @brief Get the OpenCL context
      [[nodiscard]] cl_context getContext() const noexcept
      {
        return mContext;
      }
#   endif
    protected:
    private:
#   if AFFT_GPU_FRAMEWORK_IS_CUDA
#   elif AFFT_GPU_FRAMEWORK_IS_HIP
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
      cl_context mContext; ///< OpenCL context
#   endif  
  };
} // namespace afft

#endif /* AFFT_GPU_HPP */
