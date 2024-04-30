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

#ifndef AFFT_DETAIL_UTILS_HPP
#define AFFT_DETAIL_UTILS_HPP

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <span>
#include <tuple>
#include <variant>
#include <version>

#ifndef AFFT_USE_STD_FORMAT
# include <fmt/format.h>
#elif defined(__cpp_lib_format)
# include <format>
#else
# error "std::format is not available"
#endif

#if defined(AFFT_DEBUG) && defined(__cpp_lib_source_location)
# include <source_location>
#endif

#include <mdspan.hpp>

namespace afft::detail
{
  /**
   * @brief Safely casts a value to a different integral type.
   * @tparam T Target integral type.
   * @tparam U Source integral type.
   * @param value Value to cast.
   * @return Casted value.
   * @throw std::underflow or std::overflow if the casted value is not equal to the source value.
   */
  template<std::integral T, std::integral U>
  [[nodiscard]] constexpr T safeIntCast(U value)
  {
    const auto ret = static_cast<T>(value);

    if (std::cmp_not_equal(ret, value))
    {
      if (std::cmp_less(ret, value))
      {
        throw std::underflow_error("Safe int conversion failed (underflow)");
      }
      else
      {
        throw std::overflow_error("Safe int conversion failed (overflow)");
      }
    }

    return ret;
  }

  /**
   * @brief Return result integer division.
   * @tparam I Integral type.
   */
  template<std::integral I>
  struct DivResult
  {
    I quotient;  ///< Quotient.
    I remainder; ///< Remainder.

    /**
     * @brief Converts the result to a tuple.
     * @return Tuple containing the quotient and remainder.
     */
    [[nodiscard]] constexpr operator std::tuple<I, I>() const noexcept
    {
      return std::make_tuple(quotient, remainder);
    }
  };

  /**
   * @brief Divides two integers and returns the quotient and remainder.
   * @tparam I Integral type.
   * @param a Dividend.
   * @param b Divisor.
   * @return Tuple containing the quotient and remainder.
   */
  template<std::integral I>
  [[nodiscard]] constexpr DivResult<I> div(I a, I b)
  {
    return DivResult<I>{.quotient = a / b, .remainder = a % b};
  }

  /**
   * @brief Removes the const qualifier from a pointer.
   * @tparam T Type of the pointer.
   * @param ptr Pointer to remove the const qualifier from.
   * @return Pointer without the const qualifier.
   * @warning This function should be used with caution, as it can lead to undefined behavior.
   */
  template<typename T>
  [[nodiscard]] constexpr T* removeConstFromPtr(const T* ptr)
  {
    return const_cast<T*>(ptr);
  }

inline namespace cxx20
{
  /**
   * @brief Implementation of std::format() function. If AFFT_USE_STD_FORMAT is defined, it uses std::format(),
   *        otherwise fmt::format().
   */
#ifndef AFFT_USE_STD_FORMAT
  using fmt::format;
#else
  using std::format;
#endif
} // inline namespace cxx20

// C++23 backport
inline namespace cxx23
{
  /**
   * @brief Backport of the C++23 std::to_underlying() function.
   * @tparam E Enum type.
   * @param value Enum value.
   * @return Value of the enum's underlying type.
   */
  template<typename E>
    requires std::is_enum_v<E>
  [[nodiscard]] constexpr auto to_underlying(E value) noexcept
  {
    return static_cast<std::underlying_type_t<E>>(value);
  }

  /**
   * @brief Backport of the C++23 std::unreachable() function.
   * @throw if AFFT_DEBUG is defined, otherwise calls __builtin_unreachable() or __assume(false).
   * @warning if this function ever throws, it means that there is a bug in the code, please submit an issue on GitHub.
   */
#if defined(AFFT_DEBUG) && defined(__cpp_lib_source_location)
  [[noreturn]] inline void unreachable(const std::source_location& loc = std::source_location::current())
  {
    throw std::logic_error(format("Unreachable code reached, this is a bug, please submit an issue on GitHub.\n({}:{}:{})",
                                  loc.file_name(), loc.line(), loc.column()));
  }
#else
  [[noreturn]] inline void unreachable()
  {
    // just throw now, later may be switched to real unreachable implementation
    throw std::logic_error("Unreachable code reached, this is a bug, please submit an issue on GitHub.");
// #   if defined(_MSC_VER) && !defined(__clang__)
//       __assume(false);
// #   else
//       __builtin_unreachable();
// #   endif
  }
#endif
} // inline namespace cxx23

} // afft::detail

#endif /* AFFT_DETAIL_UTILS_HPP */
