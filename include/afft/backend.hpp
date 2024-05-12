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

#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#include "macro.hpp"
#include "detail/backend.hpp"

namespace afft
{
  /// @brief Backend for the FFT
  enum class Backend : detail::BackendUnderlyingType
  {
    clfft,     ///< clFFT
    cufft,     ///< cuFFT
    fftw3,     ///< FFTW3
    hipfft,    ///< hipFFT
    mkl,       ///< Intel MKL
    pocketfft, ///< PocketFFT
    rocfft,    ///< rocFFT
    vkfft,     ///< VkFFT
  };

  /// @brief Bitmask of backends
  enum class BackendMask : detail::BackendMaskUnderlyingType
  {
    empty = detail::BackendMaskUnderlyingType{0},                          ///< empty backend mask
    all   = std::numeric_limits<detail::BackendMaskUnderlyingType>::max(), ///< all backends
  };

  /// @brief Backend select strategy
  enum class BackendSelectStrategy : std::uint8_t
  {
    first, ///< select the first available backend
    best,  ///< select the best available backend
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
} // namespace afft

#endif /* AFFT_BACKEND_HPP */
