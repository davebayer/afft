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

#ifndef AFFT_DETAIL_TYPE_HPP
#define AFFT_DETAIL_TYPE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"

namespace afft
{
  // Forward declaration
  AFFT_EXPORT template<typename>
  struct TypeProperties;

namespace detail
{
  /**
   * @brief Gets the size of a floating-point type.
   * @tparam prec The precision.
   * @tparam cmpl The complexity.
   * @return The size of the floating-point type. If the precision is not supported, returns 0.
   */
  template<Precision prec, Complexity cmpl = Complexity::real>
  [[nodiscard]] constexpr std::size_t sizeOf() noexcept
  {
    static_assert(isValid(prec), "Invalid precision.");
    static_assert(isValid(cmpl), "Invalid complexity.");

    constexpr std::size_t cmlpScale = (cmpl == Complexity::real) ? 1 : 2;

    if constexpr (prec == Precision::_longDouble)
    {
      return cmlpScale * sizeof(long double);
    }
    else
    {
      switch (prec)
      {
      case Precision::bf16:
        return cmlpScale * 2;
      case Precision::f16:
        return cmlpScale * 2;
      case Precision::f32:
        return cmlpScale * 4;
      case Precision::f64:
        return cmlpScale * 8;
      case Precision::f64f64:
        return cmlpScale * 16;
      case Precision::f80: // fixme: size may vary depending on the platform
        return cmlpScale * 16;
      case Precision::f128:
        return cmlpScale * 16;
      default:
        return 0;
      }
    }
  }
  

  /**
   * @brief Gets the size of a floating-point type.
   * @param prec The precision.
   * @return The size of the floating-point type. If the precision is not supported, returns 0.
   */
  [[nodiscard]] constexpr std::size_t sizeOf(Precision prec) noexcept
  {
    switch (prec)
    {
    case Precision::bf16:   return sizeOf<Precision::bf16>();
    case Precision::f16:    return sizeOf<Precision::f16>();
    case Precision::f32:    return sizeOf<Precision::f32>();
    case Precision::f64:    return sizeOf<Precision::f64>();
    case Precision::f64f64: return sizeOf<Precision::f64f64>();
    case Precision::f80:    return sizeOf<Precision::f80>();
    case Precision::f128:   return sizeOf<Precision::f128>();
    default:                return 0;
    }
  }
  
  /// @brief Base structure for unknown type properties.
  struct UnknownTypePropertiesBase {};

  /// @brief Base structure for known type properties.
  struct KnownTypePropertiesBase {};
} // namespace detail
} // namespace afft

#endif /* AFFT_DETAIL_TYPE_HPP */
