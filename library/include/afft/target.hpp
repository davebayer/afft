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
# include "detail/include.hpp"
#endif

#include "Span.hpp"

#ifdef AFFT_ENABLE_CUDA
# include "detail/cuda/cuda.hpp"
#endif
#ifdef AFFT_ENABLE_HIP
# include "detail/hip/hip.hpp"
#endif
#ifdef AFFT_ENABLE_OPENCL
# include "detail/opencl/opencl.hpp"
#endif

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
    static constexpr Target target = _target;
  };

  namespace cpu
  {
    struct Parameters;
    struct ExecutionParameters;
  } // namespace cpu

  namespace cuda
  {
    struct Parameters;
    struct ExecutionParameters;
  } // namespace cuda

  namespace hip
  {
    struct Parameters;
    struct ExecutionParameters;
  } // namespace hip

  namespace opencl
  {
    struct Parameters;
    struct ExecutionParameters;
  } // namespace opencl

  namespace openmp
  {
    struct Parameters;
    struct ExecutionParameters;
  } // namespace openmp

#ifdef AFFT_ENABLE_CPU
  struct cpu::Parameters : TargetConstant<Target::cpu>
  {
    unsigned threadLimit{}; ///< Thread limit for transform, 0 for no limit
  };
  
  struct cpu::ExecutionParameters : TargetConstant<Target::cpu>
  {
    void* externalWorkspace{}; ///< External workspace, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_CUDA
  struct cuda::Parameters : TargetConstant<Target::cuda>
  {
    View<int> devices{}; ///< CUDA devices
  };

  struct cuda::ExecutionParameters : TargetConstant<Target::cuda>
  {
    cudaStream_t stream{0};            ///< CUDA stream
    View<void*>  externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_HIP
  struct hip::Parameters : TargetConstant<Target::hip>
  {
    View<int> devices{}; ///< HIP devices
  };

  struct hip::ExecutionParameters : TargetConstant<Target::hip>
  {
    hipStream_t stream{0};            ///< HIP stream
    View<void*> externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
  };
#endif

#ifdef AFFT_ENABLE_OPENCL
  struct opencl::Parameters : TargetConstant<Target::opencl>
  {
    cl_context         context{}; ///< OpenCL context
    View<cl_device_id> devices{}; ///< OpenCL devices
  };

  struct opencl::ExecutionParameters : TargetConstant<Target::opencl>
  {
    cl_command_queue queue{};              ///< OpenCL command queue
    View<cl_mem>     externalWorkspaces{}; ///< External workspaces, if Workspace::external is used
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
