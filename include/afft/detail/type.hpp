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

#include "common.hpp"

namespace afft
{
  // Forward declaration
  template<typename>
  struct TypeProperties;

namespace detail
{
  /**
   * @struct FloatSelect
   * @brief Selects the floating-point type based on the precision.
   * @tparam prec The precision.
   */
  template<Precision prec>
  struct FloatSelect
  {
    using Type = void;
  };

#ifdef AFFT_HAS_BF16
  /// @brief Specialization for bf16 precision.
  template<>
  struct FloatSelect<Precision::bf16>
  {
# if defined(__GNUG__) || defined(__clang__)
    using Type = __bf16;
# else
    using Type = void;
# endif
  };
#endif /* AFFT_HAS_BF16 */

#ifdef AFFT_HAS_F16
  /// @brief Specialization for f16 precision.
  template<>
  struct FloatSelect<Precision::f16>
  {
# if defined(__GNUG__) || defined(__clang__)
    using Type = _Float16;
# else
    using Type = void;
# endif
  };
#endif /* AFFT_HAS_F16 */

  /// @brief Specialization for f32 precision.
  template<>
  struct FloatSelect<Precision::f32>
  {
    using Type = float;
  };

  /// @brief Specialization for f64 precision.
  template<>
  struct FloatSelect<Precision::f64>
  {
    using Type = double;
  };

  /// @brief Specialization for f64f64 precision.
  template<>
  struct FloatSelect<Precision::f64f64>
  {
    using Type = void;
  };

#ifdef AFFT_HAS_F80
  /// @brief Specialization for f80 precision.
  template<>
  struct FloatSelect<Precision::f80>
  {
# if defined(__GNUG__) || defined(__clang__)
    using Type = _Float64x;
# else
    using Type = void;
# endif
  };
#endif /* AFFT_HAS_F80 */

#ifdef AFFT_HAS_F128
  /// @brief Specialization for f128 precision.
  template<>
  struct FloatSelect<Precision::f128>
  {
// # if defined(__GNUG__) || defined(__clang__)
    // using Type = __float128;
// # else
    using Type = void;
// # endif
  };
#endif /* AFFT_HAS_F128 */

  /**
   * @brief Floating-point type selected according to precision.
   * @tparam prec The precision.
   */
  template<Precision prec>
    requires (isValidPrecision(prec))
  using Float = typename FloatSelect<prec>::Type;

  /**
   * @brief Checks if the given precision is supported.
   * @tparam prec The precision.
   * @return True if the precision is supported, false otherwise.
   */
  template<Precision prec>
  [[nodiscard]] inline constexpr bool hasPrecision() noexcept
  {
    return !std::same_as<detail::Float<prec>, void>;
  }

  /**
   * @brief Checks if the given precision is supported.
   * @param prec The precision.
   * @return True if the precision is supported, false otherwise.
   */
  [[nodiscard]] inline constexpr bool hasPrecision(Precision prec) noexcept
  {
    switch (prec)
    {
    case Precision::bf16:   return hasPrecision<Precision::bf16>();
    case Precision::f16:    return hasPrecision<Precision::f16>();
    case Precision::f32:    return hasPrecision<Precision::f32>();
    case Precision::f64:    return hasPrecision<Precision::f64>();
    case Precision::f64f64: return hasPrecision<Precision::f64f64>();
    case Precision::f80:    return hasPrecision<Precision::f80>();
    case Precision::f128:   return hasPrecision<Precision::f128>();
    default:                return false;
    }
  }

  /**
   * @brief Gets the size of a floating-point type.
   * @tparam prec The precision.
   * @tparam cmpl The complexity.
   * @return The size of the floating-point type. If the precision is not supported, returns 0.
   */
  template<Precision prec, Complexity cmpl = Complexity::real>
    requires (isValidPrecision(prec) && isValidComplexity(cmpl))
  [[nodiscard]] inline constexpr std::size_t sizeOf() noexcept
  {
    if constexpr (hasPrecision<prec>())
    {
      if constexpr (cmpl == Complexity::real)
      {
        return sizeof(Float<prec>);
      }
      else
      {
        return 2 * sizeof(Float<prec>);
      }
    }
    else
    {
      return std::size_t{};
    }
  }

  /**
   * @brief Gets the size of a floating-point type.
   * @param prec The precision.
   * @return The size of the floating-point type. If the precision is not supported, returns 0.
   */
  [[nodiscard]] inline constexpr std::size_t sizeOf(Precision prec) noexcept
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
