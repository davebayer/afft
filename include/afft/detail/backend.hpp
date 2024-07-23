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
#include "../common.hpp"

namespace afft
{
  // Forward declarations
  AFFT_EXPORT enum class Backend : afft_Backend;
  AFFT_EXPORT enum class BackendMask : afft_BackendMask;

namespace detail
{
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
    const auto val = detail::cxx::to_underlying(value);

    return static_cast<BackendMask>(fn(val));
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
    const auto left  = detail::cxx::to_underlying(lhs);
    const auto right = detail::cxx::to_underlying(rhs);

    return static_cast<BackendMask>(fn(left, right));
  }
} // namespace detail
} // namespace afft

#endif /* AFFT_DETAIL_BACKEND_HPP */
