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

#ifndef AFFT_BACKEND_HPP
#define AFFT_BACKEND_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "mp.hpp"
#include "target.hpp"
#include "detail/cxx.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Backend for the FFT
  enum class Backend : ::afft_Backend
  {
    clfft     = afft_Backend_clfft,     ///< clFFT
    cufft     = afft_Backend_cufft,     ///< cuFFT
    fftw3     = afft_Backend_fftw3,     ///< FFTW3
    heffte    = afft_Backend_heffte,    ///< HeFFTe
    hipfft    = afft_Backend_hipfft,    ///< hipFFT
    mkl       = afft_Backend_mkl,       ///< Intel MKL
    pocketfft = afft_Backend_pocketfft, ///< PocketFFT
    rocfft    = afft_Backend_rocfft,    ///< rocFFT
    vkfft     = afft_Backend_vkfft,     ///< VkFFT
  };

  /// @brief Number of backends
  inline constexpr std::size_t backendCount = AFFT_BACKEND_COUNT;

  /// @brief Backend constant
  template<Backend _backend>
  struct BackendConstant
  {
    static constexpr Backend backend = _backend;
  };

  /// @brief Bitmask of backends
  enum class BackendMask : ::afft_BackendMask
  {
    empty     = afft_BackendMask_empty,     ///< empty backend mask
    clfft     = afft_BackendMask_clfft,     ///< clFFT mask
    cufft     = afft_BackendMask_cufft,     ///< cuFFT mask
    fftw3     = afft_BackendMask_fftw3,     ///< FFTW3 mask
    heffte    = afft_BackendMask_heffte,    ///< HeFFTe mask
    hipfft    = afft_BackendMask_hipfft,    ///< hipFFT mask
    mkl       = afft_BackendMask_mkl,       ///< Intel MKL mask
    pocketfft = afft_BackendMask_pocketfft, ///< PocketFFT mask
    rocfft    = afft_BackendMask_rocfft,    ///< rocFFT mask
    vkfft     = afft_BackendMask_vkfft,     ///< VkFFT mask
    all       = afft_BackendMask_all,       ///< all backends
  };

  // Check that the BackendMask underlying type has sufficient size to store all Backend values
  static_assert((sizeof(::afft_BackendMask) * CHAR_BIT) >= backendCount,
                "BackendMask does not have sufficient size to store all Backend values");

  /// @brief Backend select strategy
  enum class SelectStrategy : ::afft_SelectStrategy
  {
    first = afft_SelectStrategy_first, ///< Select the first available backend
    best  = afft_SelectStrategy_best,  ///< Select the best available backend
  };

  /// @brief Workspace type
  enum class Workspace : ::afft_Workspace
  {
    any            = afft_Workspace_any,            ///< any workspace
    none           = afft_Workspace_none,           ///< no workspace
    internal       = afft_Workspace_internal,       ///< internal workspace
    external       = afft_Workspace_external,       ///< external workspace
    enlargedBuffer = afft_Workspace_enlargedBuffer, ///< enlarged buffer
  };

  /**
   * @brief Applies the bitwise `not` operation to a BackendMask or Backend.
   * @param value Value to apply the operation to.
   * @return Result of the operation.
   */
  [[nodiscard]] constexpr BackendMask operator~(BackendMask value) noexcept
  {
    const auto val = detail::cxx::to_underlying(value);

    return static_cast<BackendMask>(std::bit_not<>{}(val));
  }

  /**
   * @brief Applies the bitwise `and` operation to a BackendMask or Backend.
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  [[nodiscard]] constexpr BackendMask operator&(BackendMask lhs, BackendMask rhs) noexcept
  {
    const auto left  = detail::cxx::to_underlying(lhs);
    const auto right = detail::cxx::to_underlying(rhs);

    return static_cast<BackendMask>(std::bit_and<>{}(left, right));
  }

  /**
   * @brief Applies the bitwise `or` operation to a BackendMask or Backend.
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  [[nodiscard]] constexpr BackendMask operator|(BackendMask lhs, BackendMask rhs) noexcept
  {
    const auto left  = detail::cxx::to_underlying(lhs);
    const auto right = detail::cxx::to_underlying(rhs);

    return static_cast<BackendMask>(std::bit_or<>{}(left, right));
  }

  /**
   * @brief Applies the bitwise `xor` operation to a BackendMask or Backend.
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  [[nodiscard]] constexpr BackendMask operator^(BackendMask lhs, BackendMask rhs) noexcept
  {
    const auto left  = detail::cxx::to_underlying(lhs);
    const auto right = detail::cxx::to_underlying(rhs);

    return static_cast<BackendMask>(std::bit_xor<>{}(left, right));
  }

  /**
   * @brief Makes a BackendMask from a Backend.
   * @param backend Backend.
   * @return BackendMask.
   */
  [[nodiscard]] constexpr BackendMask makeBackendMask(Backend backend) noexcept
  {
    return static_cast<BackendMask>(1 << detail::cxx::to_underlying(backend));
  }

/**********************************************************************************************************************/
// clFFT
/**********************************************************************************************************************/
  namespace clfft
  {
    namespace opencl
    {
      struct BackendParameters;
    } // namespace opencl
  } // namespace clfft

  /// @brief clFFT OpenCL backend parameters
  struct clfft::opencl::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::opencl>, BackendConstant<Backend::clfft>
  {
    bool useFastMath{true}; ///< Use fast math.
  };

/**********************************************************************************************************************/
// cuFFT
/**********************************************************************************************************************/
  namespace cufft
  {
    enum class WorkspacePolicy : ::afft_cufft_WorkspacePolicy;

    namespace cuda
    {
      struct BackendParameters;
    } // namespace cuda

    namespace mpi::cuda
    {
      struct BackendParameters;
    } // namespace mpi::cuda
  } // namespace cufft

  /// @brief cuFFT workspace policy
  enum class cufft::WorkspacePolicy : ::afft_cufft_WorkspacePolicy
  {
    performance = afft_cufft_WorkspacePolicy_performance, ///< Use the workspace for performance
    minimal     = afft_cufft_WorkspacePolicy_minimal,     ///< Use the minimal workspace
    user        = afft_cufft_WorkspacePolicy_user,        ///< Use the user-defined workspace size
  };

  /// @brief cuFFT CUDA backend parameters
  struct cufft::cuda::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::cuda>, BackendConstant<Backend::cufft>
  {
    WorkspacePolicy    workspacePolicy{WorkspacePolicy::performance}; ///< Workspace policy.
    bool               usePatientJit{true};                           ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
    const std::size_t* userWorkspaceSize{};                           ///< Workspace size in bytes when using user-defined workspace policy.
  };

  /// @brief cuFFT MPI CUDA backend parameters
  struct cufft::mpi::cuda::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cuda>, BackendConstant<Backend::cufft>
  {
    bool usePatientJit{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };

/**********************************************************************************************************************/
// FFTW3
/**********************************************************************************************************************/
  namespace fftw3
  {
    enum class PlannerFlag : ::afft_fftw3_PlannerFlag;

    namespace cpu
    {
      struct BackendParameters;
    } // namespace cpu

    namespace mpi::cpu
    {
      struct BackendParameters;
    } // namespace mpi::cpu
  } // namespace fftw3

  /// @brief FFTW3 planner flags
  enum class fftw3::PlannerFlag : ::afft_fftw3_PlannerFlag
  {
    estimate        = afft_fftw3_PlannerFlag_estimate,        ///< Estimate plan flag
    measure         = afft_fftw3_PlannerFlag_measure,         ///< Measure plan flag
    patient         = afft_fftw3_PlannerFlag_patient,         ///< Patient plan flag
    exhaustive      = afft_fftw3_PlannerFlag_exhaustive,      ///< Exhaustive planner flag
    estimatePatient = afft_fftw3_PlannerFlag_estimatePatient, ///< Estimate and patient plan flag
  };

  /// @brief No time limit for the planner
  inline constexpr std::chrono::duration<double> noTimeLimit{AFFT_FFTW3_NO_TIME_LIMIT};

  /// @brief FFTW3 cpu backend parameters
  struct fftw3::cpu::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::cpu>, BackendConstant<Backend::fftw3>
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{noTimeLimit};             ///< Time limit for the planner
  };

  /// @brief FFTW3 MPI cpu backend parameters
  struct fftw3::mpi::cpu::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cpu>, BackendConstant<Backend::fftw3>
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{noTimeLimit};             ///< Time limit for the planner
    Size                          blockSize{};                        ///< Decomposition block size
  };

/**********************************************************************************************************************/
// HeFFTe
/**********************************************************************************************************************/
  namespace heffte
  {
    namespace cpu
    {
      enum class Backend : ::afft_heffte_cpu_Backend;
    } // namespace cpu
    namespace mpi::cpu
    {
      using heffte::cpu::Backend;
      struct BackendParameters;
    } // namespace mpi::cpu
    namespace cuda
    {
      enum class Backend : ::afft_heffte_cuda_Backend;
    } // namespace cuda
    namespace mpi::cuda
    {
      using heffte::cuda::Backend;
      struct BackendParameters;
    } // namespace mpi::cuda
    namespace hip
    {
      enum class Backend : ::afft_heffte_hip_Backend;
    } // namespace hip
    namespace mpi::hip
    {
      using heffte::hip::Backend;
      struct BackendParameters;
    } // namespace mpi::hip
  } // namespace heffte

  /// @brief HeFFTe cpu backend
  enum class heffte::cpu::Backend : ::afft_heffte_cpu_Backend
  {
    fftw3 = afft_heffte_cpu_Backend_fftw3, ///< FFTW3 backend
    mkl   = afft_heffte_cpu_Backend_mkl,   ///< MKL backend
  };

  /// @brief HeFFTe MPI cpu backend parameters
  struct heffte::mpi::cpu::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cpu>, BackendConstant<afft::Backend::heffte>
  {
    cpu::Backend backend{cpu::Backend::fftw3}; ///< Backend
    bool         useReorder{true};             ///< Use reorder flag
    bool         useAllToAll{true};            ///< Use alltoall flag
    bool         usePencils{true};             ///< Use pencils flag
  };

  /// @brief HeFFTe CUDA backend
  enum class heffte::cuda::Backend : ::afft_heffte_cuda_Backend
  {
    cufft = afft_heffte_gpu_Backend_cufft, ///< cuFFT backend
  };

  /// @brief HeFFTe MPI CUDA backend parameters
  struct heffte::mpi::cuda::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cuda>, BackendConstant<afft::Backend::heffte>
  {
    cuda::Backend backend{cuda::Backend::cufft}; ///< Backend
    bool          useReorder{true};              ///< Use reorder flag
    bool          useAllToAll{true};             ///< Use alltoall flag
    bool          usePencils{true};              ///< Use pencils flag
  };

  /// @brief HeFFTe HIP backend
  enum class heffte::hip::Backend : ::afft_heffte_hip_Backend
  {
    rocfft = afft_heffte_hip_Backend_rocfft, ///< rocFFT backend
  };

  /// @brief HeFFTe MPI HIP backend parameters
  struct heffte::mpi::hip::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::hip>, BackendConstant<afft::Backend::heffte>
  {
    hip::Backend backend{hip::Backend::rocfft}; ///< Backend
    bool         useReorder{true};              ///< Use reorder flag
    bool         useAllToAll{true};             ///< Use alltoall flag
    bool         usePencils{true};              ///< Use pencils flag
  };

/**********************************************************************************************************************/
// Backend parameters
/**********************************************************************************************************************/
  namespace cpu
  {
    namespace fftw3
    {
      using afft::fftw3::cpu::BackendParameters;
    } // namespace fftw3
    struct BackendParameters;
  } // namespace cpu
  namespace cuda
  {
    namespace cufft
    {
      using afft::cufft::cuda::BackendParameters;
    } // namespace cufft
    struct BackendParameters;
  } // namespace cuda
  namespace hip
  {
    struct BackendParameters;
  } // namespace hip
  namespace opencl
  {
    namespace clfft
    {
      using afft::clfft::opencl::BackendParameters;
    } // namespace clfft
    struct BackendParameters;
  } // namespace opencl
  namespace openmp
  {
    struct BackendParameters;
  } // namespace openmp

  /// @brief CPU backend parameters
  struct cpu::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::cpu>
  {
    SelectStrategy           strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                workspace{Workspace::any};       ///< Workspace type
    BackendMask              mask{BackendMask::all};          ///< Backend mask
    View<Backend>            order{};                         ///< Backend initialization order
    fftw3::BackendParameters fftw3{};                         ///< FFTW3 backend initialization parameters
  };

  /// @brief CUDA backend parameters
  struct cuda::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::cuda>
  {
    SelectStrategy           strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                workspace{Workspace::any};       ///< Workspace type
    BackendMask              mask{BackendMask::all};          ///< Backend mask
    View<Backend>            order{};                         ///< Backend initialization order
    cufft::BackendParameters cufft{};                         ///< cuFFT backend initialization parameters
  };

  /// @brief HIP backend parameters
  struct hip::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::hip>
  {
    SelectStrategy strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace      workspace{Workspace::any};       ///< Workspace type
    BackendMask    mask{BackendMask::all};          ///< Backend mask
    View<Backend>  order{};                         ///< Backend initialization order
  };

  /// @brief OpenCL backend parameters
  struct opencl::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::opencl>
  {
    SelectStrategy           strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                workspace{Workspace::any};       ///< Workspace type
    BackendMask              mask{BackendMask::all};          ///< Backend mask
    View<Backend>            order{};                         ///< Backend initialization order
    clfft::BackendParameters clfft{};                         ///< clFFT backend initialization parameters
  };

  /// @brief OpenMP backend parameters
  struct openmp::BackendParameters
    : MpBackendConstant<MpBackend::none>, TargetConstant<Target::openmp>
  {
    SelectStrategy strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace      workspace{Workspace::any};       ///< Workspace type
    BackendMask    mask{BackendMask::all};          ///< Backend mask
    View<Backend>  order{};                         ///< Backend initialization order
  };

  namespace mpi
  {
    namespace cpu
    {
      namespace fftw3
      {
        using afft::fftw3::mpi::cpu::BackendParameters;
      } // namespace fftw3
      namespace heffte
      {
        using afft::heffte::mpi::cpu::BackendParameters;
      } // namespace heffte
      struct BackendParameters;
    } // namespace cpu
    namespace cuda
    {
      namespace cufft
      {
        using afft::cufft::mpi::cuda::BackendParameters;
      } // namespace cufft
      namespace heffte
      {
        using afft::heffte::mpi::cuda::BackendParameters;
      } // namespace heffte
      struct BackendParameters;
    } // namespace cuda
    namespace hip
    {
      namespace heffte
      {
        using afft::heffte::mpi::hip::BackendParameters;
      } // namespace heffte
      namespace rocfft
      {
        using afft::heffte::mpi::hip::BackendParameters;
      } // namespace rocfft
      struct BackendParameters;
    } // namespace hip
    namespace opencl
    {
      struct BackendParameters;
    } //
  } // namespace mpi

  /// @brief MPI CPU backend parameters
  struct mpi::cpu::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cpu>
  {
    SelectStrategy            strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                 workspace{Workspace::any};       ///< Workspace type
    BackendMask               mask{BackendMask::all};          ///< Backend mask
    View<Backend>             order{};                         ///< Backend initialization order
    fftw3::BackendParameters  fftw3{};                         ///< FFTW3 backend initialization parameters
    heffte::BackendParameters heffte{};                        ///< HeFFTe backend initialization parameters
  };

  /// @brief MPI CUDA backend parameters
  struct mpi::cuda::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::cuda>
  {
    SelectStrategy            strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                 workspace{Workspace::any};       ///< Workspace type
    BackendMask               mask{BackendMask::all};          ///< Backend mask
    View<Backend>             order{};                         ///< Backend initialization order
    cufft::BackendParameters  cufft{};                         ///< cuFFT backend initialization parameters
    heffte::BackendParameters heffte{};                        ///< HeFFTe backend initialization parameters
  };

  /// @brief MPI HIP backend parameters
  struct mpi::hip::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::hip>
  {
    SelectStrategy            strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace                 workspace{Workspace::any};       ///< Workspace type
    BackendMask               mask{BackendMask::all};          ///< Backend mask
    View<Backend>             order{};                         ///< Backend initialization order
    heffte::BackendParameters heffte{};                        ///< HeFFTe backend initialization parameters
  };

  /// @brief MPI OpenCL backend parameters
  struct mpi::opencl::BackendParameters
    : MpBackendConstant<MpBackend::mpi>, TargetConstant<Target::opencl>
  {
    SelectStrategy strategy{SelectStrategy::first}; ///< Backend select strategy
    Workspace      workspace{Workspace::any};       ///< Workspace type
    BackendMask    mask{BackendMask::all};          ///< Backend mask
    View<Backend>  order{};                         ///< Backend initialization order
  };

  /**
   * @brief Feedback from the backend initialization.
   */
  struct Feedback
  {
    Backend                       backend{};      ///< Backend that was initialized
    std::string                   message{};      ///< Message from the backend
    std::chrono::duration<double> measuredTime{}; ///< Measured time of the backend's transformation (if available)
  };

  /**
   * @brief Get the name of the backend.
   * @param backend Backend.
   * @return Name of the backend.
   */
  [[nodiscard]] constexpr std::string_view getBackendName(Backend backend) noexcept
  {
    switch (backend)
    {
    case Backend::clfft:
      return "clFFT";
    case Backend::cufft:
      return "cuFFT";
    case Backend::fftw3:
      return "FFTW3";
    case Backend::heffte:
      return "HeFFTe";
    case Backend::hipfft:
      return "hipFFT";
    case Backend::mkl:
      return "Intel MKL";
    case Backend::pocketfft:
      return "PocketFFT";
    case Backend::rocfft:
      return "rocFFT";
    case Backend::vkfft:
      return "VkFFT";
    default:
      return "<invalid backend>";
    }
  }
} // namespace afft

#endif /* AFFT_BACKEND_HPP */
