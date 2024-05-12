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

#include <array>
#include <cstddef>

#include "backend.hpp"
#include "common.hpp"
#include "distrib.hpp"
#include "macro.hpp"
#include "mp.hpp"
#include "detail/cxx.hpp"

// Include the appropriate GPU backend
#if AFFT_GPU_BACKEND_IS(CUDA)
# include "detail/gpu/cuda/cuda.hpp"
#elif AFFT_GPU_BACKEND_IS(HIP)
# include "detail/gpu/hip/hip.hpp"
#elif AFFT_GPU_BACKEND_IS(OPENCL)
# include "detail/gpu/opencl/opencl.hpp"
#endif

namespace afft
{
namespace spst::gpu
{
  /// @brief Backend mask for single process, single target GPU target
  inline constexpr BackendMask backendMask
  {
    BackendMask::empty
# if AFFT_GPU_BACKEND_IS(CUDA)
    | Backend::cufft | Backend::vkfft
# elif AFFT_GPU_BACKEND_IS(HIP)
#   if defined(__HIP_PLATFORM_AMD__)
    | Backend::rocfft | Backend::vkfft
#   elif defined(__HIP_PLATFORM_NVIDIA__)
    | Backend::hipfft | Backend::rocfft | Backend::vkfft
#   endif
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    | Backend::clfft | Backend::vkfft
# endif
  };

  /// @brief Default backend initialization order
  inline constexpr std::array defaultBackendInitOrder = detail::cxx::to_array<Backend>(
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    Backend::cufft,  // prefer cufft
    Backend::vkfft,  // fallback to vkfft
# elif AFFT_GPU_BACKEND_IS(HIP)
#   if defined(__HIP_PLATFORM_AMD__)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
#   elif defined(__HIP_PLATFORM_NVIDIA__)
    Backend::hipfft, // prefer cufft (it is used by hipfft on CUDA)
    Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
    Backend::rocfft, // fallback to rocfft
#  endif
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    Backend::vkfft,  // prefer vkfft
    Backend::clfft,  // fallback to clfft
# endif
  });

  /**
   * @struct Parameters
   * @brief Parameters for GPU backend
   */
  struct Parameters
#if AFFT_GPU_IS_ENABLED
  {
    MemoryLayout    memoryLayout{};                                ///< Memory layout for CPU transform
    ComplexFormat   complexFormat{ComplexFormat::interleaved};     ///< complex number format
    bool            preserveSource{true};                          ///< preserve source data
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< workspace policy
# if AFFT_GPU_BACKEND_IS(CUDA)
    int             device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
# elif AFFT_GPU_BACKEND_IS(HIP)
    int             device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context      context{};                                     ///< OpenCL context
    cl_device_id    device{};                                      ///< OpenCL device
# endif
    bool            externalWorkspace{false};                      ///< Use external workspace, defaults to `false`
  }
#endif
   ;

  /**
   * @struct ExecutionParameters
   * @brief Execution parameters for gpu target
   */
  struct ExecutionParameters
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    cudaStream_t     stream{0};   ///< CUDA stream, defaults to `zero` stream
    void*            workspace{}; ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_BACKEND_IS(HIP)
    hipStream_t      stream{0};    ///< HIP stream, defaults to `zero` stream
    void*            workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_command_queue commandQueue{};
    cl_mem           workspace{};
# endif
  };
} // namespace spst

namespace spmt::gpu
{
  /// @brief Maximum number of multi devices
  inline constexpr std::size_t maxDevices{16}; ///< maximum number of devices

  /// @brief Backend mask for multi gpu target
  inline constexpr BackendMask backendMask
  {
    BackendMask::empty
#if AFFT_GPU_BACKEND_IS(CUDA)
    | Backend::cufft
#elif AFFT_GPU_BACKEND_IS(HIP)
    | Backend::hipfft | Backend::rocfft
#endif
  };

  /// @brief Order of initialization of backends
  inline constexpr std::array defaultBackendInitOrder = detail::cxx::to_array<Backend>(
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    Backend::cufft, // just cufft
# elif AFFT_GPU_BACKEND_IS(HIP)
#   if defined(__HIP_PLATFORM_AMD__)
    Backend::rocfft, // prefer rocfft
    Backend::hipfft, // fallback to hipfft
#   elif defined(__HIP_PLATFORM_NVIDIA__)
    Backend::hipfft, // prefer hipfft
    Backend::rocfft, // fallback to rocfft
#   endif
# endif
  });

  /// @brief Parameters for multi gpu target
  struct Parameters
#if AFFT_GPU_IS_ENABLED && (AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP))
  {
    MemoryLayout    memoryLayout{};                                ///< memory layout
    ComplexFormat   complexFormat{ComplexFormat::interleaved};     ///< complex number format
    bool            preserveSource{true};                          ///< preserve source data
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< workspace policy
# if AFFT_GPU_BACKEND_IS(CUDA)
    View<int>       devices{};                                     ///< list of CUDA devices
# elif AFFT_GPU_BACKEND_IS(HIP)
    View<int>       devices{};                                     ///< list of HIP devices
# endif
    bool            externalWorkspace{false};                      ///< use external workspace, defaults to `false`
  }
#endif
   ;

  /// @brief Execution parameters for multi gpu target
  struct ExecutionParameters
#if AFFT_GPU_IS_ENABLED && (AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP))
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    cudaStream_t stream{0};   ///< CUDA stream, defaults to `zero` stream
    View<void*>  workspace{}; ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_BACKEND_IS(HIP)
    hipStream_t  stream{0};    ///< HIP stream, defaults to `zero` stream
    View<void*>  workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
# endif
  }
#endif
   ;
} // namespace spmt::gpu

namespace mpst::gpu
{
  /// @brief Backend mask for multi process gpu target
  inline constexpr BackendMask backendMask
  {
    BackendMask::empty
#if AFFT_GPU_BACKEND_IS(CUDA)
    | Backend::cufft
#endif
  };

  /// @brief Order of initialization of backends
  inline constexpr std::array defaultBackendInitOrder = detail::cxx::to_array<Backend>(
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    Backend::cufft, // just cufft
# endif
  });

  /// @brief Parameters for multi process gpu target
  struct Parameters
#if AFFT_GPU_IS_ENABLED && AFFT_MP_IS_ENABLED
  {
    MemoryLayout           memoryLayout{};                                ///< memory layout
    ComplexFormat          complexFormat{ComplexFormat::interleaved};     ///< complex number format
    bool                   preserveSource{true};                          ///< preserve source data
    WorkspacePolicy        workspacePolicy{WorkspacePolicy::performance}; ///< workspace policy
    MultiProcessParameters multiProcessParameters{};                 ///< multi-process parameters
# if AFFT_GPU_BACKEND_IS_CUDA
    int                    device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
# elif AFFT_GPU_BACKEND_IS_HIP
    int                    device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
# elif AFFT_GPU_BACKEND_IS_OPENCL
    cl_context             context{};                                     ///< OpenCL context
    cl_device_id           device{};                                      ///< OpenCL device
# endif
    bool                   externalWorkspace{false};                      ///< use external workspace, defaults to `false`
  }
#endif
   ;

  /// @brief Execution parameters for mutli process gpu target
  using ExecutionParameters = std::conditional_t<(AFFT_GPU_IS_ENABLED && AFFT_MP_IS_ENABLED),
                                                 spst::gpu::ExecutionParameters,
                                                 void>;
} // namespace mpst::gpu

namespace gpu
{
  /// @brief Introduce single process, single target backend mask to the cpu namespace
  using spst::gpu::backendMask;

  /// @brief Introduce single process, single target default backend initialization order to the cpu namespace
  using spst::gpu::defaultBackendInitOrder;

  /// @brief Introduce single process, single target parameters to the cpu namespace
  using Parameters = spst::gpu::Parameters;

  /// @brief Introduce single process, single target execution parameters to the cpu namespace
  using ExecutionParameters = spst::gpu::ExecutionParameters;

  /**
   * @class UnifiedMemoryAllocator
   * @brief Allocator named concept implementation implementation for unified GPU memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
  class UnifiedMemoryAllocator
#if AFFT_GPU_IS_ENABLED
  {
    public:
      /// @brief Type of the memory
      using value_type = T;

#   if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
      /// @brief Default constructor
      constexpr UnifiedMemoryAllocator() noexcept = default;
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
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
#   if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
      : mContext(other.context)
#   endif
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr UnifiedMemoryAllocator([[maybe_unused]] UnifiedMemoryAllocator<U>&& other) noexcept
#   if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
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
#       if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
#       elif AFFT_GPU_BACKEND_IS(OPENCL)
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
#       if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
#       elif AFFT_GPU_BACKEND_IS(OPENCL)
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

#     if AFFT_GPU_BACKEND_IS(CUDA)
        detail::Error::check(cudaMallocManaged(&ptr, sizeInBytes));
#     elif AFFT_GPU_BACKEND_IS(HIP)
        detail::Error::check(hipMallocManaged(&ptr, sizeInBytes));
#     elif AFFT_GPU_BACKEND_IS(OPENCL)
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
#     if AFFT_GPU_BACKEND_IS(CUDA)
        detail::Error::check(cudaFree(p));
#     elif AFFT_GPU_BACKEND_IS(HIP)
        detail::Error::check(hipFree(p));
#     elif AFFT_GPU_BACKEND_IS(OPENCL)
        clSVMFree(mContext, p);
#     endif
      }

#   if AFFT_GPU_BACKEND_IS(OPENCL)
      /// @brief Get the OpenCL context
      [[nodiscard]] cl_context getContext() const noexcept
      {
        return mContext;
      }
#   endif
    protected:
    private:
#   if AFFT_GPU_BACKEND_IS(CUDA)
#   elif AFFT_GPU_BACKEND_IS(HIP)
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
      cl_context mContext; ///< OpenCL context
#   endif  
  }
#endif
   ;
} // namespace gpu
} // namespace afft

#endif /* AFFT_GPU_HPP */
