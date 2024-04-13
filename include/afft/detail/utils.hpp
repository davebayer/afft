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
#include <variant>
#include <version>

#ifdef __cpp_lib_format
# include <format>
#else
# include <fmt/format.h>
#endif

#include <mdspan.hpp>

namespace afft::detail
{
# ifdef __cpp_lib_format
  using std::format;
# else
  using fmt::format;
# endif

  /**
   * @brief Returns the index of a type in a variant. Inpired by:
   *        https://stackoverflow.com/questions/52303316/get-index-by-type-in-stdvariant
   * @tparam VariantType Variant type.
   * @tparam T Type to find.
   * @tparam index Index of the type in the variant.
   * @return Index of the type in the variant.
   */
  template<typename VariantType, typename T, std::size_t index = 0>
  constexpr std::size_t variant_alternative_index()
  {
    if constexpr (index >= std::variant_size_v<VariantType>)
    {
      return std::variant_npos;
    }
    else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>)
    {
      return index;
    }
    else
    {
      return variant_alternative_index<VariantType, T, index + 1>();
    }
  }

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
   * @brief Divides two integers and returns the quotient and remainder.
   * @tparam I Integral type.
   * @param a Dividend.
   * @param b Divisor.
   * @return Tuple containing the quotient and remainder.
   */
  template<std::integral I>
  [[nodiscard]] constexpr std::tuple<I, I> div(I a, I b)
  {
    return std::make_tuple(a / b, a % b);
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
  [[noreturn]] inline void unreachable()
  {
# ifdef AFFT_DEBUG
    throw std::logic_error("Unreachable code reached, this is a bug, please submit an issue on GitHub.");
# else
#   if defined(_MSC_VER) && !defined(__clang__)
      __assume(false);
#   else
      __builtin_unreachable();
#   endif
# endif
  }
}

} // afft::detail

#endif /* AFFT_DETAIL_UTILS_HPP */
