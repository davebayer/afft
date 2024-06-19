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

#ifndef AFFT_ARCHITECTURE_HPP
#define AFFT_ARCHITECTURE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "backend.hpp"
#include "detail/architecture.hpp"
#include "detail/utils.hpp"

#if AFFT_MP_BACKEND_IS(MPI)
# include "detail/mpi/mpi.hpp"
#endif

#if AFFT_GPU_BACKEND_IS(CUDA)
# include "cuda.hpp"
#elif AFFT_GPU_BACKEND_IS(HIP)
# include "hip.hpp"
#elif AFFT_GPU_BACKEND_IS(OPENCL)
# include "detail/opencl/opencl.hpp"
#endif

AFFT_EXPORT namespace afft
{
  /**
   * @brief Memory block
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt = dynamicExtent>
  struct MemoryBlock
  {
    View<std::size_t, shapeExt> starts{};  ///< starts of the memory block
    View<std::size_t, shapeExt> sizes{};   ///< sizes of the memory block
    View<std::size_t, shapeExt> strides{}; ///< strides of the memory block
  };

/**********************************************************************************************************************/
// Spst architectures
/**********************************************************************************************************************/
inline namespace spst
{
  template<std::size_t shapeExt = dynamicExtent>
  struct MemoryLayout;

  namespace cpu
  {
    template<std::size_t shapeExt = dynamicExtent>
    struct Parameters;
    struct ExecutionParameters;
  } // namespace cpu
  namespace gpu
  {
    template<std::size_t shapeExt = dynamicExtent>
    struct Parameters;
    struct ExecutionParameters;
  } // namespace gpu

  /**
   * @brief Memory layout
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct MemoryLayout
  {
    View<std::size_t, shapeExt> srcStrides{}; ///< stride of the source data
    View<std::size_t, shapeExt> dstStrides{}; ///< stride of the destination data
  };

  /**
   * @brief Parameters for spst cpu architecture
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct cpu::Parameters : detail::ArchitectureParametersBase<Target::cpu, Distribution::spst, shapeExt>
  {
    MemoryLayout<shapeExt> memoryLayout{};                            ///< Memory layout for CPU transform
    ComplexFormat          complexFormat{ComplexFormat::interleaved}; ///< complex number format
    bool                   preserveSource{true};                      ///< preserve source data
    static constexpr bool  useExternalWorkspace{false};               ///< use external workspace, disabled for now as no backend supports it
    Alignment              alignment{Alignment::defaultNew};          ///< Alignment for CPU memory allocation, defaults to `alignments::defaultNew`
    unsigned               threadLimit{};                             ///< Thread limit for CPU transform, 0 for no limit
  };

  /// @brief Execution parameters for spst cpu architecture
  struct cpu::ExecutionParameters : detail::ArchitectureExecutionParametersBase<Target::cpu, Distribution::spst>
  {};

  /**
   * @brief Parameters for spst gpu architecture
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct gpu::Parameters : detail::ArchitectureParametersBase<Target::gpu, Distribution::spst, shapeExt>
  {
    MemoryLayout<shapeExt> memoryLayout{};                            ///< Memory layout for GPU transform
    ComplexFormat          complexFormat{ComplexFormat::interleaved}; ///< complex number format
    bool                   preserveSource{true};                      ///< preserve source data
    bool                   useExternalWorkspace{false};               ///< use external workspace
# if AFFT_GPU_BACKEND_IS(CUDA)
    int                    device{cuda::getCurrentDevice()};          ///< CUDA device
# elif AFFT_GPU_BACKEND_IS(HIP)
    int                    device{hip::getCurrentDevice()};           ///< HIP device
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context             context{};                                 ///< OpenCL context
    cl_device_id           device{};                                  ///< OpenCL device
# endif
  };

  /// @brief Execution parameters for spst gpu architecture
  struct gpu::ExecutionParameters : detail::ArchitectureExecutionParametersBase<Target::gpu, Distribution::spst>
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    cudaStream_t     stream{0};   ///< CUDA stream
    void*            workspace{}; ///< workspace for spst gpu transform
# elif AFFT_GPU_BACKEND_IS(HIP)
    hipStream_t      stream{0};   ///< HIP stream
    void*            workspace{}; ///< workspace for spst gpu transform
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_command_queue queue{};     ///< OpenCL command queue
    cl_mem           workspace{}; ///< workspace for spst gpu transform
# endif
  };
} // inline namespace spst

/**********************************************************************************************************************/
// Spmt architectures
/**********************************************************************************************************************/
namespace spmt
{
  template<std::size_t shapeExt = dynamicExtent>
  struct MemoryLayout;
  
  namespace gpu
  {
    template<std::size_t shapeExt = dynamicExtent>
    struct Parameters;
    struct ExecutionParameters;
  } // namespace gpu

  /**
   * @brief Memory layout
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct MemoryLayout
  {
    View<MemoryBlock<shapeExt>> srcBlocks{};    ///< source memory blocks
    View<MemoryBlock<shapeExt>> dstBlocks{};    ///< destination memory blocks
    View<std::size_t, shapeExt> srcAxesOrder{}; ///< order of the source axes
    View<std::size_t, shapeExt> dstAxesOrder{}; ///< order of the destination axes
  };

  /**
   * @brief Parameters for spmt gpu architecture
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct gpu::Parameters : detail::ArchitectureParametersBase<Target::gpu, Distribution::spmt, shapeExt>
  {
    MemoryLayout<shapeExt> memoryLayout{};                            ///< Memory layout for GPU transform
    ComplexFormat          complexFormat{ComplexFormat::interleaved}; ///< complex number format
    bool                   preserveSource{true};                      ///< preserve source data
    bool                   useExternalWorkspace{false};               ///< use external workspace
# if AFFT_GPU_BACKEND_IS(CUDA)
    View<int>              devices{};                                 ///< CUDA devices
# elif AFFT_GPU_BACKEND_IS(HIP)
    View<int>              devices{};                                 ///< HIP devices
# endif
  };

  /// @brief Execution parameters for spmt gpu architecture
  struct gpu::ExecutionParameters : detail::ArchitectureExecutionParametersBase<Target::gpu, Distribution::spmt>
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    cudaStream_t stream{0};    ///< CUDA stream
    View<void*>  workspaces{}; ///< workspaces for spmt gpu transform
# elif AFFT_GPU_BACKEND_IS(HIP)
    hipStream_t  stream{0};    ///< HIP stream
    View<void*>  workspaces{}; ///< workspaces for spmt gpu transform
# endif
  };
} // namespace spmt

/**********************************************************************************************************************/
// Mpst architectures
/**********************************************************************************************************************/
namespace mpst
{
  template<std::size_t shapeExt = dynamicExtent>
  struct MemoryLayout;

  namespace cpu
  {
    template<std::size_t shapeExt = dynamicExtent>
    struct Parameters;
    struct ExecutionParameters;
  } // namespace cpu
  namespace gpu
  {
    template<std::size_t shapeExt = dynamicExtent>
    struct Parameters;
    struct ExecutionParameters;
  } // namespace gpu

  /**
   * @brief Memory layout
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct MemoryLayout
  {
    MemoryBlock<shapeExt>       srcBlock{};     ///< source memory block
    MemoryBlock<shapeExt>       dstBlock{};     ///< destination memory block
    View<std::size_t, shapeExt> srcAxesOrder{}; ///< order of the source axes
    View<std::size_t, shapeExt> dstAxesOrder{}; ///< order of the destination axes
  };

  /**
   * @brief Parameters for mpst cpu architecture
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct cpu::Parameters : detail::ArchitectureParametersBase<Target::cpu, Distribution::mpst, shapeExt>
  {
    MemoryLayout<shapeExt> memoryLayout{};                            ///< Memory layout for CPU transform
    ComplexFormat          complexFormat{ComplexFormat::interleaved}; ///< complex number format
    bool                   preserveSource{true};                      ///< preserve source data
    bool                   useExternalWorkspace{false};               ///< use external workspace
# if AFFT_MP_BACKEND_IS(MPI)
    MPI_Comm               communicator{MPI_COMM_WORLD};              ///< MPI communicator
# endif
    Alignment              alignment{Alignment::defaultNew};          ///< Alignment for CPU memory allocation
    unsigned               threadLimit{1};                            ///< Thread limit for CPU transform
  };

  /// @brief Execution parameters for mpst cpu architecture
  struct cpu::ExecutionParameters : detail::ArchitectureExecutionParametersBase<Target::cpu, Distribution::mpst>
  {
    void* workspace{}; ///< workspace for mpst cpu transform
  };

  /**
   * @brief Parameters for mpst gpu architecture
   * @tparam shapeExt Extent of the shape
   */
  template<std::size_t shapeExt>
  struct gpu::Parameters : detail::ArchitectureParametersBase<Target::gpu, Distribution::mpst, shapeExt>
  {
    MemoryLayout<shapeExt> memoryLayout{};                            ///< Memory layout for GPU transform
    ComplexFormat          complexFormat{ComplexFormat::interleaved}; ///< complex number format
    bool                   preserveSource{true};                      ///< preserve source data
    bool                   useExternalWorkspace{false};               ///< use external workspace
# if AFFT_MP_BACKEND_IS(MPI)
    MPI_Comm               communicator{MPI_COMM_WORLD};              ///< MPI communicator
# endif
# if AFFT_GPU_BACKEND_IS(CUDA)
    int                    device{cuda::getCurrentDevice()};          ///< CUDA device
# elif AFFT_GPU_BACKEND_IS(HIP)
    int                    device{hip::getCurrentDevice()};           ///< HIP device
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context             context{};                                 ///< OpenCL context
    cl_device_id           device{};                                  ///< OpenCL device
# endif
  };

  /// @brief Execution parameters for mpst gpu architecture
  struct gpu::ExecutionParameters : detail::ArchitectureExecutionParametersBase<Target::gpu, Distribution::mpst>
  {
# if AFFT_GPU_BACKEND_IS(CUDA)
    cudaStream_t     stream{0};   ///< CUDA stream
    void*            workspace{}; ///< workspace for mpst gpu transform
# elif AFFT_GPU_BACKEND_IS(HIP)
    hipStream_t      stream{0};   ///< HIP stream
    void*            workspace{}; ///< workspace for mpst gpu transform
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_command_queue queue{};     ///< OpenCL command queue
    cl_mem           workspace{}; ///< workspace for mpst gpu transform
# endif
  };
} // namespace mpst
} // namespace afft

#endif /* AFFT_ARCHITECTURE_HPP */
