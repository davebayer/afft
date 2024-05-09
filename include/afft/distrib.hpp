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

/// @brief MPI distribution implementation
#define AFFT_DISTRIB_IMPL_MPI    (1 << 0)

// Set default distribution implementation if not defined
#ifndef AFFT_DISTRIB_IMPL_MASK
# define AFFT_DISTRIB_IMPL_MASK  0
#endif

/**
 * @brief Check if distribution implementation is enabled
 * @param implName Distribution implementation name
 * @return non-zero if distribution implementation is enabled, zero otherwise
 */
#define AFFT_DISTRIB_IMPL_IS_ENABLED(implName) \
  (AFFT_DISTRIB_IMPL_MASK & AFFT_DISTRIB_IMPL_##implName)

// Include distribution implementation headers
#if AFFT_DISTRIB_IMPL_IS_ENABLED(MPI)
# include <mpi.h>
#endif

#include "common.hpp"
#include "cpu.hpp"
#include "gpu.hpp"

namespace afft::distrib
{
  /**
   * @enum Implementation
   * @brief Distribution implementation
   */
  enum class Implementation
  {
    none,   ///< none
    native, ///< native
    mpi,    ///< mpi
  };

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
   * @tparam impl Implementation
   * @brief Parameters for distributed CPU backend
   */
  template<Implementation impl>
  struct Parameters;

  /// @brief Parameters for undistributed CPU implementation
  template<>
  struct Parameters<Implementation::none> : afft::cpu::Parameters {};

#if AFFT_DISTRIB_IMPL_IS_ENABLED(MPI)
  /// @brief Parameters for distributed mpi CPU implementation
  template<>
  struct Parameters<Implementation::mpi>
  {
    MPI_Comm     communicator{MPI_COMM_WORLD};
    MemoryLayout memoryLayout{};
    Alignment    alignment{afft::cpu::alignments::defaultNew};
    unsigned     threadLimit{1};
  };
#endif

  /**
   * @struct ExecutionParameters
   * @tparam impl Implementation
   * @brief Execution parameters for distributed CPU backend
   */
  template<Implementation impl>
  struct ExecutionParameters : afft::cpu::ExecutionParameters {};
} // namespace cpu

namespace gpu
{
  inline constexpr std::size_t maxNativeDevices{16}; ///< Maximum number of native devices

  /**
   * @struct Parameters
   * @tparam impl Implementation
   * @brief Parameters for distributed GPU backend
   */
  template<Implementation impl>
  struct Parameters;

  /// @brief Parameters for undistributed GPU implementation
  template<>
  struct Parameters<Implementation::none> : afft::gpu::Parameters {};

  /// @brief Parameters for distributed native GPU implementation
  template<>
  struct Parameters<Implementation::native>
  {
    Span<const MemoryLayout> memoryLayouts{};          ///< span of memory layouts
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    Span<const int>          devices{};                ///< span of CUDA device IDs
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    Span<const int>          devices{};                ///< span of HIP device IDs
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context               context{};                ///< OpenCL context
    Span<const cl_device_id> devices{};                ///< span of OpenCL device IDs
# endif
    bool                     externalWorkspace{false}; ///< external workspace flag
  };

#if AFFT_DISTRIB_IMPL_IS_ENABLED(MPI)
  /// @brief Parameters for distributed mpi GPU implementation
  template<>
  struct Parameters<Implementation::mpi>
  {
    MPI_Comm     communicator{MPI_COMM_WORLD};                  ///< MPI communicator
    MemoryLayout memoryLayout{};                                ///< memory layout
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    int          device{detail::gpu::cuda::getCurrentDevice()}; ///< CUDA device ID
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    int          device{detail::gpu::hip::getCurrentDevice()};  ///< HIP device ID
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context   context{};                                     ///< OpenCL context
    cl_device_id device{};                                      ///< OpenCL device ID
# endif
    bool         externalWorkspace{false};                      ///< external workspace flag
  };
#endif

  /**
   * @struct ExecutionParameters
   * @tparam impl Implementation
   * @brief Execution parameters for distributed GPU backend
   */
  template<Implementation impl>
  struct ExecutionParameters : afft::gpu::ExecutionParameters {};

  /// @brief Execution parameters for distributed native GPU implementation
  template<>
  struct ExecutionParameters<Implementation::native>
  {
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    cudaStream_t       stream{0};    ///< CUDA stream, defaults to `zero` stream
    Span<void* const>  workspace{};  ///< span of workspace pointers, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    hipStream_t        stream{0};    ///< HIP stream, defaults to `zero` stream
    Span<void* const>  workspace{};  ///< span of workspace pointers, must be specified if `externalWorkspace` is `true`
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_command_queue   commandQueue{};
    Span<const cl_mem> workspace{};
# endif
  };
} // namespace gpu
} // namespace afft::distrib

#endif /* AFFT_DISTRIB_HPP */
