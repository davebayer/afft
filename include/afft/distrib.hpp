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

#ifndef AFFT_DISTRIB_HPP
#define AFFT_DISTRIB_HPP

/// @brief Native distribution implementation
#define AFFT_DISTRIB_IMPL_NATIVE 0

/// @brief MPI distribution implementation
#define AFFT_DISTRIB_IMPL_MPI    1

// Set default distribution implementation if not defined
#ifndef AFFT_DISTRIB_IMPL
# define AFFT_DISTRIB_IMPL       AFFT_DISTRIB_IMPL_NATIVE
#else
  // Check if distribution implementation is supported
# if (AFFT_DISTRIB_IMPL != AFFT_DISTRIB_IMPL_NATIVE) && \
     (AFFT_DISTRIB_IMPL != AFFT_DISTRIB_IMPL_MPI)
#  error "Unsupported distribution implementation"
# endif
#endif

/**
 * @brief Check if distribution implementation is `implName`
 * @param implName Distribution implementation name
 */
#define AFFT_DISTRIB_IMPL_IS(implName) \
  (AFFT_DISTRIB_IMPL == AFFT_DISTRIB_IMPL_##implName)

// Include distribution implementation headers
#if AFFT_DISTRIB_IMPL_IS(NATIVE)
#elif AFFT_DISTRIB_IMPL_IS(MPI)
# include <mpi.h>
#endif

#include "common.hpp"
#include "cpu.hpp"
#include "gpu.hpp"

namespace afft::distrib
{
  /**
   * @struct MemoryBlock
   * @brief Memory block
   */
  struct MemoryBlock
  {
    Span<const std::size_t> starts{};  ///< starts of the memory block
    Span<const std::size_t> sizes{};   ///< sizes of the memory block
    Span<const std::size_t> strides{}; ///< strides of the memory block
  };

  /**
   * @struct MemoryLayout
   * @brief Memory layout
   */
  struct MemoryLayout
  {
    MemoryBlock srcBlock{}; ///< source memory block
    MemoryBlock dstBlock{}; ///< destination memory block
  };

namespace cpu
{
  /**
   * @struct Parameters
   * @brief Parameters for distributed CPU backend
   */
  struct Parameters
  {
# if AFFT_DISTRIB_IMPL_IS(NATIVE)
# elif AFFT_DISTRIB_IMPL_IS(MPI)
    MPI_Comm     communicator{MPI_COMM_WORLD};                 ///< MPI communicator, defaults to `MPI_COMM_WORLD`
# endif
    MemoryLayout memoryLayout{};                               ///< Memory layout for distributed CPU transform
    Alignment    alignment{afft::cpu::alignments::defaultNew}; ///< Alignment for distributed CPU memory allocation
    unsigned     threadLimit{1};                               ///< Thread limit for distributed CPU transform, defaults to 1
  };

  /// @brief Execution parameters for distributed CPU transform
  struct ExecutionParameters {};
} // namespace cpu

namespace gpu
{
  inline constexpr std::size_t maxNativeDevices{16}; ///< Maximum number of native devices

  /**
   * @struct Parameters
   * @brief Parameters for distributed GPU backend
   */
  struct Parameters
  {
# if AFFT_DISTRIB_IMPL_IS(NATIVE)
    Span<const MemoryLayout> memoryLayouts{};                               ///< Memory layout for distributed GPU transform
  // GPU framework specific parameters
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    Span<const int>          devices{};                                     ///< CUDA devices, defaults to current device
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    Span<const int>          devices{};                                     ///< HIP devices, defaults to current device
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context               context{};                                     ///< OpenCL context
    Span<const cl_device_id> devices{};                                     ///< OpenCL devices
# endif
# elif AFFT_DISTRIB_IMPL_IS(MPI)
    MPI_Comm                 communicator{MPI_COMM_WORLD};                  ///< MPI communicator, defaults to `MPI_COMM_WORLD`
    MemoryLayout             memoryLayout{};                                ///< Memory layout for distributed GPU transform
  // GPU framework specific parameters
#   if AFFT_GPU_FRAMEWORK_IS_CUDA
    int                      device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device, defaults to current device
#   elif AFFT_GPU_FRAMEWORK_IS_HIP
    int                      device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device, defaults to current device
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context               context{};                                     ///< OpenCL context
    cl_device_id             device{};                                      ///< OpenCL device
#   endif
# endif
    bool                     externalWorkspace{false};                      ///< Use external workspace, defaults to `false`
  };

  /**
   * @struct ExecutionParameters
   * @brief Execution parameters for distributed gpu target
   */
  struct ExecutionParameters
  {
# if AFFT_DISTRIB_IMPL_IS(NATIVE)
  // GPU framework specific execution parameters
#   if AFFT_GPU_FRAMEWORK_IS_CUDA
    cudaStream_t       stream{0};    ///< CUDA stream, defaults to `zero` stream
    Span<void* const>  workspace{};  ///< span of workspace pointers, must be specified if `externalWorkspace` is `true`
#   elif AFFT_GPU_FRAMEWORK_IS_HIP
    hipStream_t        stream{0};    ///< HIP stream, defaults to `zero` stream
    Span<void* const>  workspace{};  ///< span of workspace pointers, must be specified if `externalWorkspace` is `true`
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_command_queue   commandQueue{};
    Span<const cl_mem> workspace{};
#   endif
# elif AFFT_DISTRIB_IMPL_IS(MPI)
  // GPU framework specific execution parameters
#   if AFFT_GPU_FRAMEWORK_IS_CUDA
    cudaStream_t       stream{0};    ///< CUDA stream, defaults to `zero` stream
    void*              workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
#   elif AFFT_GPU_FRAMEWORK_IS_HIP
    hipStream_t        stream{0};    ///< HIP stream, defaults to `zero` stream
    void*              workspace{};  ///< workspace memory pointer, must be specified if `externalWorkspace` is `true`
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_command_queue   commandQueue{};
    cl_mem             workspace{};
#   endif
# endif
  };
} // namespace gpu
} // namespace afft::distrib

#endif /* AFFT_DISTRIB_HPP */
