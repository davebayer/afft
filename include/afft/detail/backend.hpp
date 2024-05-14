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

#ifndef AFFT_DETAIL_BACKEND_HPP
#define AFFT_DETAIL_BACKEND_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "cxx.hpp"

namespace afft
{
namespace detail
{
  /// @brief Underlying type of the Backend enum
  using BackendUnderlyingType = std::uint8_t;

  // Check that the Backend underlying type is unsigned
  static_assert(std::is_unsigned_v<BackendUnderlyingType>);
  
  /// @brief Underlying type of the BackendMask enum
  using BackendMaskUnderlyingType = std::uint16_t;

  // Check that the BackendMask underlying type is unsigned
  static_assert(std::is_unsigned_v<BackendMaskUnderlyingType>);

  /**
   * @brief Checks if the BackendMask underlying type has sufficient size to store all Backend values.
   * @return True if the BackendMask underlying type has sufficient size, false otherwise.
   */
  [[nodiscard]] inline constexpr bool backendMaskHasSufficientUnderlyingTypeSize(Backend backendCount)
  {
    return (sizeof(BackendMaskUnderlyingType) * CHAR_BIT) >= cxx::to_underlying(backendCount);
  }
} // namespace detail

  // Forward declarations
  AFFT_EXPORT enum class Backend : detail::BackendUnderlyingType;
  AFFT_EXPORT enum class BackendMask : detail::BackendMaskUnderlyingType;

namespace detail
{
  /**
   * @brief Converts Backend or BackednMask to a BackendMask.
   * @tparam T Type of the value.
   * @param value Value to convert.
   * @return BackendMask representation of the value.
   */
  template<typename T>
  [[nodiscard]] constexpr BackendMask toBackendMask(T value)
  {
    static_assert(std::is_same_v<T, Backend> || std::is_same_v<T, BackendMask>,
                  "T must be either Backend or BackendMask");

    if constexpr (std::is_same_v<T, Backend>)
    {
      return static_cast<BackendMask>(U{1} << detail::cxx::to_underlying(value));
    }
    else
    {
      return value;
    }
  }

  /**
   * @brief Applies a unary operation to a BackendMask.
   * @tparam UnOp Type of the unary operation.
   * @tparam T Type of the value.
   * @param fn Unary operation to apply.
   * @param value Value to apply the operation to.
   * @return Result of the operation.
   */
  template<typename UnOp, typename T>
  [[nodiscard]] constexpr BackendMask backendMaskUnaryOp(UnOp fn, T value)
  {
    const auto val = detail::cxx::to_underlying(detail::toBackendMask(value));

    return BackendMask{fn(val)};
  }

  /**
   * @brief Applies a binary operation to two BackendMask values.
   * @tparam BinFn Type of the binary operation.
   * @tparam T Type of the left-hand side value.
   * @tparam U Type of the right-hand side value.
   * @param fn Binary operation to apply.
   * @param lhs Left-hand side value.
   * @param rhs Right-hand side value.
   * @return Result of the operation.
   */
  template<typename BinFn, typename T, typename U>
  [[nodiscard]] constexpr BackendMask backendMaskBinaryOp(BinFn fn, T lhs, U rhs)
  {
    const auto left  = detail::cxx::to_underlying(detail::toBackendMask(lhs));
    const auto right = detail::cxx::to_underlying(detail::toBackendMask(rhs));

    return BackendMask{fn(left, right)};
  }
} // namespace detail
} // namespace afft

#endif /* AFFT_DETAIL_BACKEND_HPP */
