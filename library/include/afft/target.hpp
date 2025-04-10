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

#ifndef AFFT_TARGET_HPP
#define AFFT_TARGET_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.hpp>
#endif

#ifdef AFFT_ENABLE_CUDA
# include <afft/detail/cuda/cuda.hpp>
#endif
#ifdef AFFT_ENABLE_HIP
# include <afft/detail/hip/hip.hpp>
#endif
#ifdef AFFT_ENABLE_OPENCL
# include <afft/detail/opencl/opencl.hpp>
#endif

#include <afft/detail/target.hpp>

AFFT_EXPORT namespace afft
{
  /// @brief Target
  enum class Target : ::afft_Target
  {
    cpu    = afft_Target_cpu,    ///< native CPU target
    cuda   = afft_Target_cuda,   ///< CUDA target
    hip    = afft_Target_hip,    ///< HIP target
    opencl = afft_Target_opencl, ///< OpenCL target
    openmp = afft_Target_openmp, ///< OpenMP target
  };
  
  /**
   * @brief Target constant
   * @tparam _target Target
   */
  template<Target _target>
  struct TargetConstant
  {
    static constexpr Target target = _target; ///< Target
  };

  namespace cpu
  {
    /// @brief CPU parameters
    struct Parameters;

    /// @brief CPU execution parameters
    struct ExecutionParameters;
  } // namespace cpu

  namespace cuda
  {
    /// @brief CUDA parameters
    struct Parameters;

    /// @brief CUDA execution parameters
    struct ExecutionParameters;
  } // namespace cuda

  namespace hip
  {
    /// @brief HIP parameters
    struct Parameters;

    /// @brief HIP execution parameters
    struct ExecutionParameters;
  } // namespace hip

  namespace opencl
  {
    /// @brief OpenCL parameters
    struct Parameters;

    /// @brief OpenCL execution parameters
    struct ExecutionParameters;
  } // namespace opencl

  namespace openmp
  {
    /// @brief OpenMP parameters
    struct Parameters;

    /// @brief OpenMP execution parameters
    struct ExecutionParameters;
  } // namespace openmp

#ifdef AFFT_ENABLE_CPU
  /// @brief CPU parameters
  struct cpu::Parameters : TargetConstant<Target::cpu>
  {
    static constexpr std::size_t targetCount{1}; ///< Target count
  };
  
  /// @brief CPU execution parameters
  struct cpu::ExecutionParameters : TargetConstant<Target::cpu>
  {
    void* externalWorkspace{}; ///< External workspace, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_CUDA
  /// @brief CUDA parameters
  struct cuda::Parameters : TargetConstant<Target::cuda>
  {
    std::size_t targetCount{1}; ///< Target count
    const int*  devices{};      ///< CUDA devices
  };

  /// @brief CUDA execution parameters
  struct cuda::ExecutionParameters : TargetConstant<Target::cuda>
  {
    cudaStream_t stream{0};            ///< CUDA stream
    void* const* externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_HIP
  /// @brief HIP parameters
  struct hip::Parameters : TargetConstant<Target::hip>
  {
    std::size_t targetCount{1}; ///< Target count
    const int*  devices{};      ///< HIP devices
  };

  /// @brief HIP execution parameters
  struct hip::ExecutionParameters : TargetConstant<Target::hip>
  {
    hipStream_t  stream{0};            ///< HIP stream
    void* const* externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_OPENCL
  /// @brief OpenCL parameters
  struct opencl::Parameters : TargetConstant<Target::opencl>
  {
    std::size_t         targetCount{}; ///< Target count
    cl_context          context{};     ///< OpenCL context
    const cl_device_id* devices{};     ///< OpenCL devices
  };

  /// @brief OpenCL execution parameters
  struct opencl::ExecutionParameters : TargetConstant<Target::opencl>
  {
    cl_command_queue queue{};              ///< OpenCL command queue
    const cl_mem*    externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_OPENMP
  /// @brief OpenMP parameters
  struct openmp::Parameters : TargetConstant<Target::openmp>
  {
    int device{}; ///< OpenMP device
  };

  /// @brief OpenMP execution parameters
  struct openmp::ExecutionParameters : TargetConstant<Target::openmp>
  {
    bool nowait{}; ///< Nowait
  };
#endif

  /// @brief Target parameters variant
  using TargetParametersVariant = std::variant<
    std::monostate
# ifdef AFFT_ENABLE_CPU
  , cpu::Parameters
# endif
# ifdef AFFT_ENABLE_CUDA
  , cuda::Parameters
# endif
# ifdef AFFT_ENABLE_HIP
  , hip::Parameters
# endif
# ifdef AFFT_ENABLE_OPENCL
  , opencl::Parameters
# endif
# ifdef AFFT_ENABLE_OPENMP
  , openmp::Parameters
# endif
  >;

  /// @brief Target execution parameters variant
  using TargetExecutionParametersVariant = std::variant<
    std::monostate
# ifdef AFFT_ENABLE_CPU
  , cpu::ExecutionParameters
# endif
# ifdef AFFT_ENABLE_CUDA
  , cuda::ExecutionParameters
# endif
# ifdef AFFT_ENABLE_HIP
  , hip::ExecutionParameters
# endif
# ifdef AFFT_ENABLE_OPENCL
  , opencl::ExecutionParameters
# endif
# ifdef AFFT_ENABLE_OPENMP
  , openmp::ExecutionParameters
# endif
  >;
} // namespace afft

#endif /* AFFT_TARGET_HPP */
