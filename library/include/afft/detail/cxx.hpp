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

#ifndef AFFT_DETAIL_CXX_HPP
#define AFFT_DETAIL_CXX_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

namespace afft::detail::cxx
{
// implementation of optinal c++11 features
inline namespace cxx11
{
#ifdef UINTPTR_MAX
  using ::std::uintptr_t;
#else
  static_assert((sizeof(void*) > 0 &&                             // void* is greater than 0
                 ((sizeof(void*) & (sizeof(void*) - 1)) == 0) &&  // void* is a power of 2
                 sizeof(void*) <= sizeof(std::uint64_t)),         // void* is not greater than 64 bits
                "uintptr_t is not defined and cannot be implemented");

  using uintptr_t = std::conditional_t<sizeof(void*) == sizeof(std::uint16_t), std::uint16_t,
                    std::conditional_t<sizeof(void*) == sizeof(std::uint32_t), std::uint32_t,
                    std::conditional_t<sizeof(void*) == sizeof(std::uint64_t), std::uint64_t, void>>>;
#endif
} // namespace cxx11

inline namespace cxx20
{
  /**
   * @brief TypeProperties helper. Removes const and volatile from Complex template parameter type.
   * @tparam T The type.
   */
  template<typename T>
  struct remove_cvref
  {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  /// @brief Shortcut for remove_cvref type.
  template<typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;

  /**
   * @brief Is bounded array type.
   * @tparam T The type.
   */
  template<typename T>
  struct is_bounded_array : std::false_type {};

  /// @brief Specialization for bounded array type.
  template<typename T, std::size_t size>
  struct is_bounded_array<T[size]> : std::true_type {};

  /**
   * @brief Shortcut for is_bounded_array value.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

  /**
   * @brief Is unbounded array type.
   * @tparam T The type.
   */
  template<typename T>
  struct is_unbounded_array : std::false_type {};

  /// @brief Specialization for unbounded array type.
  template<typename T>
  struct is_unbounded_array<T[]> : std::true_type {};

  /**
   * @brief Shortcut for is_unbounded_array value.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;

  /**
   * @brief Compares two values for equality. Taken from https://en.cppreference.com/w/cpp/utility/intcmp
   * @tparam T First value type.
   * @tparam U Second value type.
   * @param t First value.
   * @param u Second value.
   * @return true if the values are equal, false otherwise.
   */
  template<class T, class U>
  [[nodiscard]] constexpr bool cmp_equal(T t, U u) noexcept
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
  [[nodiscard]] constexpr bool cmp_not_equal(T t, U u) noexcept
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
  [[nodiscard]] constexpr bool cmp_less(T t, U u) noexcept
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
  [[nodiscard]] constexpr bool cmp_greater(T t, U u) noexcept
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
  [[nodiscard]] constexpr bool cmp_less_equal(T t, U u) noexcept
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
  [[nodiscard]] constexpr bool cmp_greater_equal(T t, U u) noexcept
  {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integral types");

    return !cmp_less(t, u);
  }

  /**
   * @brief Checks if the given value is a power of two. Taken from https://en.cppreference.com/w/cpp/numeric/has_single_bit
   * @tparam T Type of the value.
   * @param x Value to check.
   * @return true if the value is a power of two, false otherwise.
   */
  template<typename T>
  [[nodiscard]] constexpr bool has_single_bit(T x) noexcept
  {
    static_assert(std::is_unsigned_v<T>, "T must be an unsigned integral type.");
    static_assert(!std::is_same_v<T, bool>, "T must not be a boolean type.");
    static_assert(!std::is_same_v<T, char> &&
                  !std::is_same_v<T, char16_t> &&
                  !std::is_same_v<T, char32_t> &&
                  !std::is_same_v<T, wchar_t>, "T must not be a character type.");

    return x && !(x & (x - 1));
  }

  /**
   * @brief Finds the first element in the range [first, last) for which the predicate p returns true. Taken from https://en.cppreference.com/w/cpp/algorithm/find
   * @tparam InputIt Iterator type.
   * @tparam UnaryPred Predicate type.
   * @param first The beginning of the range.
   * @param last The end of the range.
   * @param p Unary predicate which returns true for the required element. Default is std::equal_to<>.
   * @return Iterator to the first element for which the predicate returns true, or last if no such element is found.
   */
  template<class InputIt, class UnaryPred>
  constexpr InputIt find_if(InputIt first, InputIt last, UnaryPred p = std::equal_to<>{})
  {
    for (; first != last; ++first)
    {
      if (p(*first))
      {
        return first;
      }
    }

    return last;
  }

  /**
   * @brief Finds the first element in the range [first, last) for which the predicate p returns false. Taken from https://en.cppreference.com/w/cpp/algorithm/find
   * @tparam InputIt Iterator type.
   * @tparam UnaryPred Predicate type.
   * @param first The beginning of the range.
   * @param last The end of the range.
   * @param q Unary predicate which returns false for the required element.
   * @return Iterator to the first element for which the predicate returns false, or last if no such element is found.
   */
  template<class InputIt, class UnaryPred>
  constexpr InputIt find_if_not(InputIt first, InputIt last, UnaryPred q)
  {
    for (; first != last; ++first)
    {
      if (!q(*first))
      {
        return first;
      }
    }

    return last;
  }  

  /**
   * @brief Checks if all elements in the range [first, last) satisfy the predicate p. Taken from https://en.cppreference.com/w/cpp/algorithm/all_of
   * @tparam InputIt Iterator type.
   * @tparam UnaryPred Predicate type.
   * @param first The beginning of the range.
   * @param last The end of the range.
   * @param p Unary predicate which returns true for the required element. Default is std::equal_to<>.
   * @return true if all elements in the range satisfy the predicate, false otherwise.
   */
  template<class InputIt, class UnaryPred>
  constexpr bool all_of(InputIt first, InputIt last, UnaryPred p)
  {
    return cxx::find_if_not(first, last, p) == last;
  }

  /**
   * @brief Checks if any element in the range [first, last) satisfies the predicate p. Taken from https://en.cppreference.com/w/cpp/algorithm/any_of
   * @tparam InputIt Iterator type.
   * @tparam UnaryPred Predicate type.
   * @param first The beginning of the range.
   * @param last The end of the range.
   * @param p Unary predicate which returns true for the required element. Default is std::equal_to<>.
   * @return true if any element in the range satisfies the predicate, false otherwise.
   */
  template<class InputIt, class UnaryPred>
  constexpr bool any_of(InputIt first, InputIt last, UnaryPred p = std::equal_to<>{})
  {
    return cxx::find_if(first, last, p) != last;
  }

  /**
   * @brief Checks if no element in the range [first, last) satisfies the predicate p. Taken from https://en.cppreference.com/w/cpp/algorithm/none_of
   * @tparam InputIt Iterator type.
   * @tparam UnaryPred Predicate type.
   * @param first The beginning of the range.
   * @param last The end of the range.
   * @param p Unary predicate which returns true for the required element. Default is std::equal_to<>.
   * @return true if no element in the range satisfies the predicate, false otherwise.
   */
  template<class InputIt, class UnaryPred>
  constexpr bool none_of(InputIt first, InputIt last, UnaryPred p)
  {
    return cxx::find_if(first, last, p) == last;
  }
} // namespace cxx20

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
  [[nodiscard]] constexpr std::underlying_type_t<E> to_underlying(E value) noexcept
  {
    static_assert(std::is_enum_v<E>, "to_underlying() can only be used with enum types.");

    return static_cast<std::underlying_type_t<E>>(value);
  }

  /**
   * @brief Backport of the C++23 std::unreachable() function.
   * @throw if AFFT_DEBUG is defined, otherwise calls __builtin_unreachable() or __assume(false).
   * @warning if this function ever throws, it means that there is a bug in the code, please submit an issue on GitHub.
   */
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
} // inline namespace cxx23
} // namespace afft::detail::cxx

#endif /* AFFT_DETAIL_CXX_HPP */
