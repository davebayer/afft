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
#include "../3rdparty.hpp"
#include "../Span.hpp"

namespace afft::detail
{
  /**
   * @brief Optional reference.
   * @tparam T Type of the reference.
   */
  template<typename T>
  class OptionalRef
  {
    public:
      /// @brief Default constructor.
      constexpr OptionalRef() noexcept = default;

      /**
       * @brief Constructor.
       * @param ref Reference to store.
       */
      constexpr OptionalRef(T& ref) noexcept
      : mPtr{&ref}
      {}

      /**
       * @brief Copy constructor.
       * @param other Other instance to copy.
       */
      constexpr OptionalRef(const OptionalRef& other) noexcept = default;

      /**
       * @brief Move constructor.
       * @param other Other instance to move.
       */
      constexpr OptionalRef(OptionalRef&& other) noexcept = default;

      /// @brief Destructor.
      ~OptionalRef() noexcept = default;

      /**
       * @brief Copy assignment operator.
       * @param other Other instance to copy.
       * @return Reference to this instance.
       */
      constexpr OptionalRef& operator=(const OptionalRef& other) noexcept = default;

      /**
       * @brief Move assignment operator.
       * @param other Other instance to move.
       * @return Reference to this instance.
       */
      constexpr OptionalRef& operator=(OptionalRef&& other) noexcept = default;

      /**
       * @brief Gets the value.
       * @return Reference to the value.
       * @throw std::runtime_error if the optional does not have a value.
       */
      [[nodiscard]] constexpr T& value() const noexcept
      {
        if (!hasValue())
        {
          throw std::runtime_error("OptionalRef does not have a value");
        }

        return *mPtr;
      }

      /**
       * @brief Checks if the optional has a value.
       * @return True if the optional has a value, false otherwise.
       */
      [[nodiscard]] constexpr bool hasValue() const noexcept
      {
        return (mPtr != nullptr);
      }

      /**
       * @brief Resets the optional.
       */
      constexpr void reset() noexcept
      {
        mPtr = nullptr;
      }

      /**
       * @brief Gets the value without checking if it exists.
       * @return Reference to the value.
       */
      [[nodiscard]] constexpr T& operator*() const noexcept
      {
        return *mPtr;
      }

      /**
       * @brief Gets the value without checking if it exists.
       * @return Pointer to the value.
       */
      [[nodiscard]] constexpr T* operator->() const noexcept
      {
        return mPtr;
      }

      /**
       * @brief Checks if the optional has a value.
       * @return True if the optional has a value, false otherwise.
       */
      [[nodiscard]] constexpr operator bool() const noexcept
      {
        return hasValue();
      }
    protected:
    private:
      T* mPtr{}; ///< Pointer to the value.
  };

  /**
   * @struct IsZero
   * @brief Function object that checks if a value is zero.
   * @tparam T Type of the value.
   */
  template<typename T>
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
  template<typename T>
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
} // namespace afft::detail

#endif /* AFFT_DETAIL_UTILS_HPP */
