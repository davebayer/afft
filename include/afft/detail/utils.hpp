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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "cxx.hpp"
#include "../Span.hpp"

namespace afft::detail
{
  /**
   * @brief Creates a view over a single element.
   * @tparam T Type of the element.
   * @tparam extent Number of elements in the view.
   * @param value Element to create the view from.
   * @return View containing the element.
   */
  template<typename T, std::size_t extent = dynamicExtent>
  [[nodiscard]] constexpr View<T, extent> makeView(const T& value) noexcept
  {
    return View<T, extent>{&value, 1};
  }

  /**
   * @brief Creates an array from list of arguments
   * @tparam T Resulting array type.
   * @tparam Args Types of the arguments.
   * @param args Arguments to create the array from.
   * @return Array containing the arguments.
   */
  template<typename T, typename... Args>
  [[nodiscard]] constexpr std::array<T, sizeof...(Args)> makeArray(Args&&... args)
  {
    static_assert(std::conjunction_v<std::is_convertible_v<Args, T>...>,
                  "Arguments must be convertible to the array type");

    return std::array<T, sizeof...(Args)>{std::forward<Args>(args)...};
  }

  /**
   * @brief Reinterprets a span of elements as a span of a different type.
   * @tparam T Target type.
   * @tparam U Source type.
   * @tparam extent Number of elements in the span.
   * @param span Span to reinterpret.
   * @return Reinterpreted span.
   */
  template<typename T, typename U, std::size_t extent>
  View<T, extent> reinterpretViewCast(View<U, extent> span)
  {
    return View<T, extent>{reinterpret_cast<const T*>(span.data()), span.size()};
  }

  /**
   * @struct IsZero
   * @brief Function object that checks if a value is zero.
   * @tparam T Type of the value.
   */
  template<typename T = void>
  struct IsZero
  {
    static_assert(std::is_arithmetic_v<T>, "IsZero can only be used with arithmetic types.");

    /**
     * @brief Checks if a value is zero.
     * @param value Value to check.
     * @return True if the value is zero, false otherwise.
     */
    [[nodiscard]] constexpr bool operator()(const T& value) const noexcept
    {
      return (value == T{});
    }
  };

  /// @brief Specialization for void type. Allows to use IsZero<void> with any type.
  template<>
  struct IsZero<void>
  {
    /**
     * @brief Checks if a value is zero.
     * @tparam T Type of the value.
     * @param value Value to check.
     * @return True if the value is zero, false otherwise.
     */
    template<typename T>
    [[nodiscard]] constexpr bool operator()(T&& value) const noexcept
    {
      return IsZero<T>{}(std::forward<T>(value));
    }
  };

  /**
   * @struct IsNotZero
   * @brief Function object that checks if a value is not zero.
   * @tparam T Type of the value.
   */
  template<typename T = void>
  struct IsNotZero
  {
    static_assert(std::is_arithmetic_v<T>, "IsNotZero can only be used with arithmetic types.");

    /**
     * @brief Checks if a value is not zero.
     * @param value Value to check.
     * @return True if the value is not zero, false otherwise.
     */
    [[nodiscard]] constexpr bool operator()(const T& value) const noexcept
    {
      return (value != T{});
    }
  };

  /// @brief Specialization for void type. Allows to use IsNotZero<void> with any type.
  template<>
  struct IsNotZero<void>
  {
    /**
     * @brief Checks if a value is not zero.
     * @tparam T Type of the value.
     * @param value Value to check.
     * @return True if the value is not zero, false otherwise.
     */
    template<typename T>
    [[nodiscard]] constexpr bool operator()(T&& value) const noexcept
    {
      return IsNotZero<T>{}(std::forward<T>(value));
    }
  };

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

    if (cxx::cmp_not_equal(ret, value))
    {
      if (cxx::cmp_less(ret, value))
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
  template<typename... Args>
  [[nodiscard]] std::string cformat(std::string_view format, const Args&... args)
  {
    const auto size = std::snprintf(nullptr, 0, format.data(), args...);

    if (size == 0)
    {
      return std::string{};
    }
    else if (size > 0)
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
   * @brief Formats a string using C-style format without throwing exceptions.
   * @param format Format string.
   * @param args Arguments to format.
   * @return Formatted string.
   */
  template<typename... Args>
  [[nodiscard]] std::string cformatNothrow(std::string_view format, const Args&... args) noexcept
  {
    std::string result{};

    const auto size = std::snprintf(nullptr, 0, format.data(), args...);

    if (size > 0)
    {
      result.resize(static_cast<std::size_t>(size));

      std::snprintf(result.data(), result.size() + 1, format.data(), args...);
    }

    return result;
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
} // namespace afft::detail

#endif /* AFFT_DETAIL_UTILS_HPP */
