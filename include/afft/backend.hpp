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

#include "detail/backend.hpp"
#include "detail/utils.hpp"

#include "common.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Backend for the FFT
  enum class Backend : detail::BackendUnderlyingType
  {
    clfft,     ///< clFFT
    cufft,     ///< cuFFT
    fftw3,     ///< FFTW3
    heffte,    ///< HeFFTe
    hipfft,    ///< hipFFT
    mkl,       ///< Intel MKL
    pocketfft, ///< PocketFFT
    rocfft,    ///< rocFFT
    vkfft,     ///< VkFFT
    _count,    ///< number of backends, do not use, only for internal purposes
  };

  /// @brief Bitmask of backends
  enum class BackendMask : detail::BackendMaskUnderlyingType
  {
    empty = detail::BackendMaskUnderlyingType{0},                          ///< empty backend mask
    all   = std::numeric_limits<detail::BackendMaskUnderlyingType>::max(), ///< all backends
  };

  // Check that the BackendMask underlying type has sufficient size to store all Backend values
  static_assert(detail::backendMaskHasSufficientUnderlyingTypeSize(Backend::_count),
                "BackendMask does not have sufficient size to store all Backend values");

  /// @brief Backend select strategy
  enum class SelectStrategy : std::uint8_t
  {
    first, ///< Select the first available backend
    best,  ///< Select the best available backend
  };

  /**
   * @brief Applies the bitwise `not` operation to a BackendMask or Backend.
   * @tparam T Type of the value (Backend or BackendMask).
   * @param value Value to apply the operation to.
   * @return Result of the operation.
   */
  template<typename T>
  [[nodiscard]] inline constexpr auto operator~(T value)
    -> AFFT_RET_REQUIRES(BackendMask, (std::is_same_v<T, Backend> || std::is_same_v<T, BackendMask>))
  {
    return detail::backendMaskUnaryOp(std::bit_not<>{}, value);
  }

  /**
   * @brief Applies the bitwise `and` operation to a BackendMask or Backend.
   * @tparam T Type of the left-hand side (Backend or BackendMask).
   * @tparam U Type of the right-hand side (Backend or BackendMask).
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  template<typename T, typename U>
  [[nodiscard]] inline constexpr auto operator&(T lhs, U rhs)
    -> AFFT_RET_REQUIRES(BackendMask, (std::is_same_v<T, Backend> || std::is_same_v<T, BackendMask>) &&
                                      (std::is_same_v<U, Backend> || std::is_same_v<U, BackendMask>))
  {
    return detail::backendMaskBinaryOp(std::bit_and<>{}, lhs, rhs);
  }

  /**
   * @brief Applies the bitwise `or` operation to a BackendMask or Backend.
   * @tparam T Type of the left-hand side (Backend or BackendMask).
   * @tparam U Type of the right-hand side (Backend or BackendMask).
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  template<typename T, typename U>
  [[nodiscard]] inline constexpr auto operator|(T lhs, U rhs)
    -> AFFT_RET_REQUIRES(BackendMask, (std::is_same_v<T, Backend> || std::is_same_v<T, BackendMask>) &&
                                      (std::is_same_v<U, Backend> || std::is_same_v<U, BackendMask>))
  {
    return detail::backendMaskBinaryOp(std::bit_or<>{}, lhs, rhs);
  }

  /**
   * @brief Applies the bitwise `xor` operation to a BackendMask or Backend.
   * @tparam T Type of the left-hand side (Backend or BackendMask).
   * @tparam U Type of the right-hand side (Backend or BackendMask).
   * @param lhs Left-hand side of the operation.
   * @param rhs Right-hand side of the operation.
   * @return Result of the operation.
   */
  template<typename T, typename U>
  [[nodiscard]] inline constexpr auto operator^(T lhs, U rhs)
    -> AFFT_RET_REQUIRES(BackendMask, (std::is_same_v<T, Backend> || std::is_same_v<T, BackendMask>) &&
                                      (std::is_same_v<U, Backend> || std::is_same_v<U, BackendMask>))
  {
    return detail::backendMaskBinaryOp(std::bit_xor<>{}, lhs, rhs);
  }

/**********************************************************************************************************************/
// clFFT
/**********************************************************************************************************************/
namespace clfft::spst::gpu
{
  struct Parameters;
} // namespace clfft::spst::gpu

  /// @brief clFFT initialization parameters
  struct clfft::spst::gpu::Parameters
  {
    bool useFastMath{true}; ///< Use fast math.
  };

/**********************************************************************************************************************/
// cuFFT
/**********************************************************************************************************************/
namespace cufft
{
  enum class WorkspacePolicy : std::uint8_t;
  namespace spst::gpu
  {
    struct Parameters;
  } // namespace spst::gpu
  namespace spmt::gpu
  {
    struct Parameters;
  } // namespace spmt::gpu
  namespace mpst::gpu
  {
    struct Parameters;
  } // namespace mpst::gpu
} // namespace cufft

  /// @brief cuFFT Workspace policy
  enum class cufft::WorkspacePolicy : std::uint8_t
  {
    performance, ///< Use the workspace for performance
    minimal,     ///< Use the minimal workspace
    user,        ///< Use the user-defined workspace size
  };

  /// @brief cuFFT initialization parameters for the spst gpu architecture
  struct cufft::spst::gpu::Parameters
  {
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< Workspace policy.
    bool            usePatientJit{true};                           ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
    std::size_t     userWorkspaceSize{};                           ///< Workspace size in bytes when using user-defined workspace policy.
  };

  /// @brief cuFFT initialization parameters for the spmt gpu architecture
  struct cufft::spmt::gpu::Parameters
  {
    bool usePatientJit{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };

  /// @brief cuFFT initialization parameters for the mpst gpu architecture
  struct cufft::mpst::gpu::Parameters
  {
    bool usePatientJit{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };

/**********************************************************************************************************************/
// FFTW3
/**********************************************************************************************************************/
namespace fftw3
{
  enum class PlannerFlag : std::uint8_t;
  namespace spst::cpu
  {
    struct Parameters;
  } // namespace cpu
  namespace mpst::cpu
  {
    struct Parameters;
  } // namespace cpu
} // namespace fftw3

  /// @brief FFTW3 planner flags
  enum class fftw3::PlannerFlag : std::uint8_t
  {
    estimate,        ///< Estimate plan flag
    measure,         ///< Measure plan flag
    patient,         ///< Patient plan flag
    exhaustive,      ///< Exhaustive planner flag
    estimatePatient, ///< Estimate and patient plan flag
  };

  /// @brief FFTW3 initialization parameters for the spst cpu architecture
  struct fftw3::spst::cpu::Parameters
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{};                        ///< Time limit for the planner
  };

  /// @brief FFTW3 initialization parameters for the mpst cpu architecture
  struct fftw3::mpst::cpu::Parameters
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{};                        ///< Time limit for the planner
    std::size_t                   blockSize{};                        ///< Decomposition block size
  };

/**********************************************************************************************************************/
// HeFFTe
/**********************************************************************************************************************/
namespace heffte
{
  namespace cpu
  {
    enum class Backend : std::uint8_t;
  } // namespace cpu
  namespace gpu
  {
    enum class Backend : std::uint8_t;
  } // namespace gpu
  namespace mpst::cpu
  {
    struct Parameters;
  } // namespace mpst::cpu
  namespace mpst::gpu
  {
    struct Parameters;
  } // namespace mpst::gpu
} // namespace heffte

  /// @brief HeFFTe cpu backends
  enum class heffte::cpu::Backend : std::uint8_t
  {
    fftw3,  ///< FFTW3 backend
    mkl,    ///< Intel MKL backend
  };

  /// @brief HeFFTe gpu backends
  enum class heffte::gpu::Backend : std::uint8_t
  {
    cufft,  ///< cuFFT backend
    rocfft, ///< rocFFT backend
  };

  /// @brief HeFFTe initialization parameters for the mpst cpu architecture
  struct heffte::mpst::cpu::Parameters
  {
    using Backend = heffte::cpu::Backend;

    [[nodiscard]] static Parameters makeDefault(Backend backend)
    {
      switch (backend)
      {
#     ifdef Heffte_ENABLE_FFTW
      case Backend::fftw3:
      {
        const auto options = ::heffte::default_options<::heffte::backend::fftw>();
        return Parameters{Backend::fftw3, options.use_reorder, options.use_alltoall, options.use_pencils};
      }
#     endif
#     ifdef Heffte_ENABLE_MKL
      case Backend::mkl:
      {
        const auto options = ::heffte::default_options<::heffte::backend::mkl>();
        return Parameters{Backend::mkl, options.use_reorder, options.use_alltoall, options.use_pencils};
      }
#     endif
      default:
        return Parameters{};
      }
    }

    Backend backend{};         ///< Backend
    bool    useReorder{true};  ///< Use reorder flag
    bool    useAlltoAll{true}; ///< Use alltoall flag
    bool    usePencils{true};  ///< Use pencils flag
  };

  /// @brief HeFFTe initialization parameters for the mpst gpu architecture
  struct heffte::mpst::gpu::Parameters
  {
    using Backend = heffte::gpu::Backend;

    [[nodiscard]] static Parameters makeDefault(Backend backend)
    {
      switch (backend)
      {
#     ifdef Heffte_ENABLE_CUDA
      case Backend::cufft:
      {
        const auto options = ::heffte::default_options<::heffte::backend::cufft>();
        return Parameters{Backend::cufft, options.use_reorder, options.use_alltoall, options.use_pencils};
      }
#     endif
#     ifdef Heffte_ENABLE_ROCM
      case Backend::rocfft:
      {
        const auto options = ::heffte::default_options<::heffte::backend::rocfft>();
        return Parameters{Backend::rocfft, options.use_reorder, options.use_alltoall, options.use_pencils};
      }
#     endif
      default:
        return Parameters{};
      }
    }

    Backend backend{};         ///< Backend
    bool    useReorder{true};  ///< Use reorder flag
    bool    useAlltoAll{true}; ///< Use alltoall flag
    bool    usePencils{true};  ///< Use pencils flag
  };

/**********************************************************************************************************************/
// Backend parameters for spst distribution
/**********************************************************************************************************************/
inline namespace spst
{
  namespace cpu
  {
    namespace fftw3
    {
      using afft::fftw3::spst::cpu::Parameters;
    } // namespace fftw3

    /// @brief Supported backends for spst cpu architecture
    inline constexpr BackendMask supportedBackendMask = Backend::fftw3 |
                                                        Backend::mkl |
                                                        Backend::pocketfft;

    /// @brief Default backend order for spst cpu architecture
    inline constexpr std::array defaultBackendOrder = detail::makeArray<Backend>(Backend::mkl,
                                                                                 Backend::fftw3,
                                                                                 Backend::pocketfft);

    struct BackendParameters;
  } // namespace cpu
  namespace gpu
  {
    namespace clfft
    {
      using afft::clfft::spst::gpu::Parameters;
    } // namespace clfft
    namespace cufft
    {
      using afft::cufft::spst::gpu::Parameters;
    } // namespace cufft

    /// @brief Supported backends for spst gpu architecture
    inline constexpr BackendMask supportedBackendMask = Backend::clfft |
                                                        Backend::cufft |
                                                        Backend::hipfft |
                                                        Backend::rocfft |
                                                        Backend::vkfft;

    /// @brief Default backend order for spst gpu architecture
    inline constexpr std::array defaultBackendOrder = detail::makeArray<Backend>(
#   if AFFT_GPU_BACKEND_IS(CUDA)
      Backend::cufft,  // prefer cufft
      Backend::vkfft   // fallback to vkfft
#   elif AFFT_GPU_BACKEND_IS(HIP)
#     if defined(__HIP_PLATFORM_AMD__)
      Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
      Backend::rocfft  // fallback to rocfft
#     elif defined(__HIP_PLATFORM_NVIDIA__)
      Backend::hipfft, // prefer cufft (it is used by hipfft on CUDA)
      Backend::vkfft,  // prefer vkfft as it should be faster than rocfft
      Backend::rocfft  // fallback to rocfft
#     endif
#   elif AFFT_GPU_BACKEND_IS(OPENCL)
      Backend::vkfft,  // prefer vkfft
      Backend::clfft   // fallback to clfft
#   endif
    );

    struct BackendParameters;
  } // namespace gpu
} // namespace spst

  /// @brief Backend parameters for the spst distribution on the CPU
  struct spst::cpu::BackendParameters : detail::BackendParametersBase<Target::cpu, Distribution::spst>
  {
    SelectStrategy    strategy{SelectStrategy::first}; ///< Backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< Backend mask
    View<Backend>     order{defaultBackendOrder};      ///< Backend initialization order, empty view means default order for the target
    fftw3::Parameters fftw3{};                         ///< FFTW3 backend initialization parameters
  };

  /// @brief Backend parameters for the spst distribution on the GPU
  struct spst::gpu::BackendParameters : detail::BackendParametersBase<Target::gpu, Distribution::spst>
  {
    SelectStrategy    strategy{SelectStrategy::first}; ///< Backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< Backend mask
    View<Backend>     order{defaultBackendOrder};      ///< Backend initialization order, empty view means default order for the target
    clfft::Parameters clfft{};                         ///< clFFT backend initialization parameters
    cufft::Parameters cufft{};                         ///< cuFFT backend initialization parameters
  };

/**********************************************************************************************************************/
// Backend parameters for spmt distribution
/**********************************************************************************************************************/
namespace spmt
{
  namespace gpu
  {
    namespace cufft
    {
      using afft::cufft::spmt::gpu::Parameters;
    } // namespace cufft

    /// @brief Supported backends for spmt gpu architecture
    inline constexpr BackendMask supportedBackendMask = Backend::cufft |
                                                        Backend::hipfft |
                                                        Backend::rocfft;

    /// @brief Order of initialization of backends
    inline constexpr std::array defaultBackendOrder = detail::makeArray<Backend>(
#   if AFFT_GPU_BACKEND_IS(CUDA)
      Backend::cufft  // just cufft
#   elif AFFT_GPU_BACKEND_IS(HIP)
#     if defined(__HIP_PLATFORM_AMD__)
      Backend::rocfft, // prefer rocfft
      Backend::hipfft  // fallback to hipfft
#     elif defined(__HIP_PLATFORM_NVIDIA__)
      Backend::hipfft, // prefer hipfft
      Backend::rocfft  // fallback to rocfft
#     endif
#   endif
    );

    struct BackendParameters;
  } // namespace gpu
} // namespace spmt


  /// @brief Backend parameters for the spmt distribution on the GPU
  struct spmt::gpu::BackendParameters : detail::BackendParametersBase<Target::gpu, Distribution::spmt>
  {
    SelectStrategy    strategy{SelectStrategy::first}; ///< Backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< Backend mask
    View<Backend>     order{defaultBackendOrder};      ///< Backend initialization order, empty view means default order for the target
    cufft::Parameters cufft{};                         ///< cuFFT backend initialization parameters
  };

/**********************************************************************************************************************/
// Backend parameters for mpst distribution
/**********************************************************************************************************************/
namespace mpst
{
  namespace cpu
  {
    namespace fftw3
    {
      using afft::fftw3::mpst::cpu::Parameters;
    } // namespace fftw3
    namespace heffte
    {
      using afft::heffte::mpst::cpu::Parameters;
    } // namespace heffte

    /// @brief Supported backends for mpst cpu transform
    inline constexpr BackendMask supportedBackendMask = Backend::fftw3 |
                                                        Backend::mkl |
                                                        Backend::heffte;
    /// @brief Default backend initialization order for mpst cpu transform
    inline constexpr std::array defaultBackendOrder = detail::makeArray<Backend>(
      Backend::mkl,    // prefer mkl
      Backend::heffte, // if mkl cannot create plan, fallback heffte
      Backend::fftw3   // if heffte cannot create plan, fallback fftw3
    );

    struct BackendParameters;
  } // namespace cpu
  namespace gpu
  {
    namespace cufft
    {
      using afft::cufft::mpst::gpu::Parameters;
    } // namespace cufft
    namespace heffte
    {
      using afft::heffte::mpst::gpu::Parameters;
    } // namespace heffte

    /// @brief Supported backends for mpst gpu transform
    inline constexpr BackendMask supportedBackendMask = Backend::cufft |
                                                        Backend::heffte;

    /// @brief Order of initialization of backends
    inline constexpr std::array defaultBackendOrder = detail::makeArray<Backend>(
#   if AFFT_GPU_BACKEND_IS(CUDA)
      Backend::cufft, // try cufft first
      Backend::heffte // fallback to heffte
#   elif AFFT_GPU_BACKEND_IS(HIP)
      Backend::heffte
#   endif
    );
    struct BackendParameters;
  } // namespace gpu
} // namespace mpst

  /// @brief Backend parameters for the mpst distribution on the CPU
  struct mpst::cpu::BackendParameters : detail::BackendParametersBase<Target::cpu, Distribution::mpst>
  {
    SelectStrategy     strategy{SelectStrategy::first}; ///< Backend select strategy
    BackendMask        mask{supportedBackendMask};      ///< Backend mask
    View<Backend>      order{defaultBackendOrder};      ///< Backend initialization order, empty view means default order for the target
    fftw3::Parameters  fftw3{};                         ///< FFTW3 backend initialization parameters
    heffte::Parameters heffte{};                        ///< HeFFTe backend initialization parameters
  };

  /// @brief Backend parameters for the mpst distribution on the GPU
  struct mpst::gpu::BackendParameters : detail::BackendParametersBase<Target::gpu, Distribution::mpst>
  {
    SelectStrategy     strategy{SelectStrategy::first}; ///< Backend select strategy
    BackendMask        mask{supportedBackendMask};      ///< Backend mask
    View<Backend>      order{defaultBackendOrder};      ///< Backend initialization order, empty view means default order for the target
    cufft::Parameters  cufft{};                         ///< cuFFT backend initialization parameters
    heffte::Parameters heffte{};                        ///< HeFFTe backend initialization parameters
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
   * @brief Converts a Backend to a string.
   * @param backend Backend to convert.
   * @return String representation of the backend.
   */
  [[nodiscard]] inline constexpr std::string_view toString(Backend backend)
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
