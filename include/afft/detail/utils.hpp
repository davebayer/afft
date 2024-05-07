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

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <tuple>
#include <variant>
#include <version>

#if defined(AFFT_DEBUG) && defined(__cpp_lib_source_location)
# include <source_location>
#endif

#include "../3rdparty.hpp"
#include "../Span.hpp"

namespace afft::detail
{
inline namespace cxx20
{
  /**
   * @brief Compares two values for equality. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the values are equal, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_equal(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
    {
      return t == u;
    }
    else if constexpr (std::is_signed_v<T>)
    {
      return t >= 0 && std::make_unsigned_t<T>(t) == u;
    }
    else
    {
      return u >= 0 && std::make_unsigned_t<U>(u) == t;
    }
  }
  
  /**
   * @brief Compares two values for inequality. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the values are not equal, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_not_equal(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    return !cmp_equal(t, u);
  }
  
  /**
   * @brief Compares two values for less than. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the first value is less than the second value, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_less(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
    {
      return t < u;
    }
    else if constexpr (std::is_signed_v<T>)
    {
      return t < 0 || std::make_unsigned_t<T>(t) < u;
    }
    else
    {
      return u >= 0 && t < std::make_unsigned_t<U>(u);
    }
  }
  
  /**
   * @brief Compares two values for greater than. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the first value is greater than the second value, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_greater(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    return cmp_less(u, t);
  }
  
  /**
   * @brief Compares two values for less than or equal. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the first value is less than or equal to the second value, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_less_equal(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    return !cmp_less(u, t);
  }
  
  /**
   * @brief Compares two values for greater than or equal. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the first value is greater than or equal to the second value, false otherwise.
   */
  template<class T, class U>
  constexpr bool cmp_greater_equal(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    return !cmp_less(t, u);
  }
} // inline namespace cxx20
  /**
   * @brief Safely casts a value to a different integral type.
   * @tparam T Target integral type.
   * @tparam U Source integral type.
   * @param value Value to cast.
   * @return Casted value.
   * @throw std::underflow or std::overflow if the casted value is not equal to the source value.
   */
  template<typename T, typename U>
  [[nodiscard]] constexpr T safeIntCast(U value)
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    const auto ret = static_cast<T>(value);

    if (cmp_not_equal(ret, value))
    {
      if (cmp_less(ret, value))
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
   * @brief Formats a string using C-style format.
   * @param format Format string.
   * @param args Arguments to format.
   * @return Formatted string.
   * @throw std::runtime_error if the string could not be formatted.
   */
  [[nodiscard]] std::string cformat(std::string_view format, const auto&... args)
  {
    const auto size = std::snprintf(nullptr, 0, format.data(), args...);

    if (size >= 0)
    {
      std::string result(static_cast<std::size_t>(size), '\0');

      if (std::snprintf(result.data(), result.size() + 1, format.data(), args...) == size)
      {
        return result;
      }
    }

    throw std::runtime_error("Failed to format string");
  }

  /**
   * @brief Return result integer division.
   * @tparam I Integral type.
   */
  template<typename I>
  struct DivResult
  {
    static_assert(std::is_integral_v<I>, "DivResult can only be used with integral types.");

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
  template<typename I>
  [[nodiscard]] constexpr DivResult<I> div(I a, I b)
  {
    static_assert(std::is_integral_v<I>, "div() can only be used with integral types.");

    return DivResult<I>{/* .quotient  = */ a / b,
                        /* .remainder = */ a % b};
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
  [[nodiscard]] constexpr auto to_underlying(E value) noexcept
  {
    static_assert(std::is_enum_v<E>, "to_underlying() can only be used with enum types.");

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
    throw std::logic_error(cformat("Unreachable code reached, this is a bug, please submit an issue on GitHub.\n"
                                   "(%s:" PRIuLEAST32 ":" PRIuLEAST32 ")",
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
