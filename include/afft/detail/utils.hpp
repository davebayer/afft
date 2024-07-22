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

#include "common.hpp"
#include "cxx.hpp"
#include "../Span.hpp"

namespace afft::detail
{
  /**
   * @brief Function object that casts a value to a different type.
   * @tparam T Type of the destination value.
   */
  template<typename T>
  struct StaticCaster
  {
    /**
     * @brief Casts a value to a different type.
     * @tparam U Type of the source value.
     * @param value Value to cast.
     * @return Casted value.
     */
    template<typename U>
    [[nodiscard]] constexpr T operator()(const U& value) const
    {
      static_assert(std::is_convertible_v<U, T>, "Cannot cast value to destination type");

      return static_cast<T>(value);
    }
  };

  /**
   * @brief Function object that safely casts a value to a different integral type.
   * @tparam T Type of the destination value.
   */
  template<typename T>
  struct SafeIntCaster
  {
    /**
     * @brief Safely casts a value to a different integral type.
     * @tparam U Type of the source value.
     * @param value Value to cast.
     * @return Casted value.
     * @throw std::underflow_error if the casted value is less than the source value.
     * @throw std::overflow_error if the casted value is greater than the source value.
     */
    template<typename U>
    [[nodiscard]] constexpr T operator()(const U& value) const
    {
      static_assert(std::is_integral_v<T> && std::is_integral_v<U>,
                    "SafeIntCaster can only be used with integer types");

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
  };

  /**
   * @brief Casts a view of values to a buffer of a different type.
   * @tparam SrcIt Type of the source iterator.
   * @tparam DstIt Type of the destination iterator.
   * @tparam CastFnT Type of the casting function.
   * @param first Iterator to the first element of the source view.
   * @param last Iterator to the last element of the source view.
   * @param dest Iterator to the first element of the destination buffer.
   * @param fn Casting function.
   */
  template<typename SrcIt,
           typename DstIt,
           typename CastFnT = StaticCaster<typename std::iterator_traits<DstIt>::value_type>>
  constexpr void cast(SrcIt first, SrcIt last, DstIt dest, CastFnT&& fn = {})
  {
    std::transform(first, last, dest, std::forward<CastFnT>(fn));
  }

  /**
   * @brief Casts a view of values to a buffer of a different type.
   * @tparam DstT Type of the destination buffer.
   * @tparam SrcT Type of the source view.
   * @tparam size Size of the view and buffer.
   * @tparam CastFnT Type of the casting function.
   * @param view View of values to cast.
   * @param fn Casting function.
   * @return Buffer of casted values.
   */
  template<typename DstT,
           typename SrcT,
           std::size_t size,
           typename CastFnT = StaticCaster<DstT>>
  [[nodiscard]] constexpr auto cast(View<SrcT, size> view, CastFnT&& fn = {})
    noexcept(std::is_nothrow_invocable_r_v<DstT, CastFnT, SrcT>)
    -> AFFT_RET_REQUIRES(AFFT_PARAM(Buffer<DstT, size>),
                         AFFT_PARAM(std::is_default_constructible_v<DstT> &&
                                    size != dynamicExtent &&
                                    std::is_invocable_r_v<DstT, CastFnT, SrcT>))
  {
    Buffer<DstT, size> buffer{};

    cast(view.begin(), view.end(), buffer.data, std::forward<CastFnT>(fn));

    return buffer;
  }

  /**
   * @struct IsZero
   * @brief Function object that checks if a value is zero.
   * @tparam T Type of the value.
   */
  template<typename T = void>
  struct IsZero
  {
    static_assert(std::is_arithmetic_v<T> || std::is_void_v<T>, "IsZero can only be used with arithmetic types.");

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
    static_assert(std::is_arithmetic_v<T> || std::is_void_v<T>, "IsNotZero can only be used with arithmetic types.");

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
    return SafeIntCaster<T>{}(value);
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
   * @brief Struct that contains the NEmbed and stride format.
   * @tparam I Integral type.
   */
  template<typename I>
  struct NEmbedAndStride
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");
    
    MaxDimBuffer<I> nembed; ///< NEmbed buffer.
    I               stride; ///< Stride.

    /**
     * @brief Converts the object to a tuple.
     * @return Tuple containing the NEmbed and stride
     */
    [[nodiscard]] constexpr operator std::tuple<const MaxDimBuffer<I>&, I>() const noexcept
    {
      return std::make_tuple(nembed, stride);
    }
  };

  /**
   * @brief Creates an NEmbed and stride object from shape and strides.
   * @tparam I Integral type.
   * @tparam shapeExt Size of the shape view.
   * @tparam strideExt Size of the strides view.
   * @param shape Shape view.
   * @param strides Strides view.
   * @return NEmbed and stride object.
   * @throw std::runtime_error if the shape and strides have different sizes.
   */
  template<typename I, std::size_t shapeExt, std::size_t strideExt>
  [[nodiscard]] constexpr NEmbedAndStride<I> makeNEmbedAndStride(View<Size, shapeExt> shape, View<I, strideExt> strides)
  {
    static_assert((shapeExt == dynamicExtent) ||
                  (strideExt == dynamicExtent) ||
                  (shapeExt == strideExt), "Shape and strides must have the same size");

    if (shape.size() != strides.size())
    {
      throw std::runtime_error{"Shape and strides must have the same size"};
    }

    NEmbedAndStride<I> nembedAndStride{};

    const std::size_t rank = shape.size();

    nembedAndStride.stride = safeIntCast<I>(strides.back());

    for (std::size_t i{rank - 1}; i > 0; --i)
    {
      const auto [nembed, remainder] = div(strides[i - 1], strides[i]);

      if (nembed < strides[i] || remainder != 0)
      {
        throw std::runtime_error{"Strides cannot be converted to nembed and stride"};
      }

      nembedAndStride.nembed[i] = safeIntCast<I>(nembed);
    }

    return nembedAndStride;
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_UTILS_HPP */
