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

  /// @brief Backend select strategy
  enum class SelectStrategy : std::uint8_t
  {
    first, ///< select the first available backend
    best,  ///< select the best available backend
  };

namespace cufft
{
  /// @brief cuFFT Workspace policy
  enum class WorkspacePolicy : std::uint8_t
  {
    performance, ///< Use the workspace for performance.
    minimal,     ///< Use the minimal workspace.
    user,        ///< Use the user-defined workspace size.
  };
} // namespace cufft

namespace fftw3
{
  /// @brief FFTW3 planner flags
  enum class PlannerFlag : std::uint8_t
  {
    estimate,        ///< Estimate plan flag
    measure,         ///< Measure plan flag
    patient,         ///< Patient plan flag
    exhaustive,      ///< Exhaustive planner flag
    estimatePatient, ///< Estimate and patient plan flag
  };
} // namespace fftw3

namespace heffte
{
  enum class Backend : std::uint8_t
  {
    cufft,  ///< cuFFT backend
    fftw3,  ///< FFTW3 backend
    mkl,    ///< Intel MKL backend
    rocfft, ///< rocFFT backend
  };

  struct Parameters
  {
    Backend backend{};         ///< Backend
    bool    useReorder{true};  ///< Use reorder flag
    bool    useAlltoAll{true}; ///< Use alltoall flag
    bool    usePencils{true};  ///< Use pencils flag
  };

  inline constexpr Backend defaultBackend = Backend::fftw3;

  [[nodiscard]] Parameters makeDefaultParameters(Backend heffteBackend = defaultBackend)
  {
    switch (heffteBackend)
    {
#   ifdef Heffte_ENABLE_CUDA
    case Backend::cufft:
    {
      const auto options = ::heffte::default_options<::heffte::backend::cufft>();
      return Parameters{Backend::cufft, options.use_reorder, options.use_alltoall, options.use_pencils};
    }
#   endif
#   ifdef Heffte_ENABLE_FFTW
    case Backend::fftw3:
    {
      const auto options = ::heffte::default_options<::heffte::backend::fftw>();
      return Parameters{Backend::fftw3, options.use_reorder, options.use_alltoall, options.use_pencils};
    }
#   endif
#   ifdef Heffte_ENABLE_MKL
    case Backend::mkl:
    {
      const auto options = ::heffte::default_options<::heffte::backend::mkl>();
      return Parameters{Backend::mkl, options.use_reorder, options.use_alltoall, options.use_pencils};
    }
#   endif
#   ifdef Heffte_ENABLE_ROCM
    case Backend::rocfft:
    {
      const auto options = ::heffte::default_options<::heffte::backend::rocfft>();
      return Parameters{Backend::rocfft, options.use_reorder, options.use_alltoall, options.use_pencils};
    }
#   endif
    default:
      return Parameters{};
    }
  }
} // namespace heffte

inline namespace spst
{
namespace cpu
{
namespace fftw3
{
  using namespace afft::fftw3;

  /**
   * @brief Initialization parameters for the FFTW3 plan.
   */
  struct Parameters
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{};                        ///< Time limit for the planner
  };
} // namespace fftw3

  /// @brief Supported backends
  inline constexpr BackendMask supportedBackendMask = Backend::fftw3 |
                                                      Backend::mkl |
                                                      Backend::pocketfft;

  /// @brief Backend selection parameters
  struct BackendParameters
  {
    static constexpr Target       target{Target::cpu};              ///< target
    static constexpr Distribution distribution{Distribution::spst}; ///< distribution

    SelectStrategy    strategy{SelectStrategy::first}; ///< backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< backend mask
    View<Backend>     order{};                         ///< backend initialization order, empty view means default order for the target
    fftw3::Parameters fftw3{};                         ///< FFTW3 backend initialization parameters
  };
} // namespace cpu

namespace gpu
{
namespace clfft
{
  /// @brief clFFT initialization parameters
  struct Parameters
  {
    bool useFastMath{true}; ///< Use fast math.
  };
} // namespace clfft

namespace cufft
{
  using namespace afft::cufft;

  /// @brief cuFFT initialization parameters
  struct Parameters
  {
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< Workspace policy.
    bool            usePatientJIT{true};                           ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
    std::size_t     userWorkspaceSize{};                           ///< Workspace size in bytes when using user-defined workspace policy.
  };
} // namespace cufft

  /// @brief Supported backends
  inline constexpr BackendMask supportedBackendMask = Backend::clfft |
                                                      Backend::cufft |
                                                      Backend::hipfft |
                                                      Backend::rocfft |
                                                      Backend::vkfft;

  /// @brief Backend selection parameters
  struct BackendParameters
  {
    static constexpr Target       target{Target::gpu};              ///< target
    static constexpr Distribution distribution{Distribution::spst}; ///< distribution
    
    SelectStrategy    strategy{SelectStrategy::first}; ///< backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< backend mask
    View<Backend>     order{};                         ///< backend initialization order, empty view means default order for the target
    clfft::Parameters clfft{};                         ///< clFFT backend initialization parameters
    cufft::Parameters cufft{};                         ///< cuFFT backend initialization parameters
  };
} // namespace gpu
} // inline namespace spst

namespace spmt
{
namespace gpu
{
namespace cufft
{
  /// @brief cuFFT initialization parameters
  struct Parameters
  {
    bool usePatientJIT{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };
} // namespace cufft

  /// @brief Supported backends
  inline constexpr BackendMask supportedBackendMask = Backend::cufft |
                                                      Backend::hipfft |
                                                      Backend::rocfft;

  /// @brief Backend selection parameters
  struct BackendParameters
  {
    static constexpr Target       target{Target::gpu};              ///< target
    static constexpr Distribution distribution{Distribution::spmt}; ///< distribution

    SelectStrategy    strategy{SelectStrategy::first}; ///< backend select strategy
    BackendMask       mask{supportedBackendMask};      ///< backend mask
    View<Backend>     order{};                         ///< backend initialization order, empty view means default order for the target
    cufft::Parameters cufft{};                         ///< cuFFT backend initialization parameters
  };
} // namespace gpu
} // namespace spmt

namespace mpst
{
namespace cpu
{
namespace fftw3
{
  using namespace afft::fftw3;

  /// @brief Initialization parameters for the FFTW3 MPI plan.
  struct Parameters
  {
    PlannerFlag                   plannerFlag{PlannerFlag::estimate}; ///< FFTW3 planner flag
    bool                          conserveMemory{false};              ///< Conserve memory flag
    bool                          wisdomOnly{false};                  ///< Wisdom only flag
    bool                          allowLargeGeneric{false};           ///< Allow large generic flag
    bool                          allowPruning{false};                ///< Allow pruning flag
    std::chrono::duration<double> timeLimit{};                        ///< Time limit for the planner
    std::size_t                   blockSize{};                        ///< Decomposition block size
  };
} // namespace fftw3
namespace heffte
{
  /// @brief HeFFTe initialization parameters
  using afft::heffte::Parameters;
} // namespace heffte

  /// @brief Supported backends
  inline constexpr BackendMask supportedBackendMask = Backend::fftw3 |
                                                      Backend::heffte |
                                                      Backend::mkl;

  /// @brief Backend selection parameters
  struct BackendParameters
  {
    static constexpr Target       target{Target::cpu};              ///< target
    static constexpr Distribution distribution{Distribution::mpst}; ///< distribution

    SelectStrategy     strategy{SelectStrategy::first};               ///< backend select strategy
    BackendMask        mask{supportedBackendMask};                    ///< backend mask
    View<Backend>      order{};                                       ///< backend initialization order, empty view means default order for the target
    fftw3::Parameters  fftw3{};                                       ///< FFTW3 backend initialization parameters
    heffte::Parameters heffte{afft::heffte::makeDefaultParameters()}; ///< HeFFTe backend initialization parameters
  };
} // namespace cpu

namespace gpu
{
namespace cufft
{
  /// @brief cuFFT initialization parameters
  struct Parameters
  {
    bool usePatientJIT{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };
} // namespace cufft
namespace heffte
{
  /// @brief HeFFTe initialization parameters
  using afft::heffte::Parameters;
} // namespace heffte

  /// @brief Supported backends
  inline constexpr BackendMask supportedBackendMask = Backend::cufft |
                                                      Backend::heffte;

  /// @brief Backend selection parameters
  struct BackendParameters
  {
    static constexpr Target       target{Target::gpu};              ///< target
    static constexpr Distribution distribution{Distribution::mpst}; ///< distribution

    SelectStrategy     strategy{SelectStrategy::first};               ///< backend select strategy
    BackendMask        mask{supportedBackendMask};                    ///< backend mask
    View<Backend>      order{};                                       ///< backend initialization order, empty view means default order for the target
    cufft::Parameters  cufft{};                                       ///< cuFFT backend initialization parameters
    heffte::Parameters heffte{afft::heffte::makeDefaultParameters()}; ///< HeFFTe backend initialization parameters
  };
} // namespace gpu
} // namespace mpst

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
