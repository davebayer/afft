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

#ifndef AFFT_SPAN_HPP
#define AFFT_SPAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

AFFT_EXPORT namespace afft
{
#ifdef AFFT_CXX_HAS_SPAN
  /// @brief The dynamic extent value for Span.
  inline constexpr std::size_t dynamicExtent = std::dynamic_extent;

  /// @brief The Span type.
  template<typename T, std::size_t extent = dynamicExtent>
  using Span = std::span<T, extent>;
#else
  /// @brief The dynamic extent value for Span.
  inline constexpr std::size_t dynamicExtent = afft::thirdparty::span::dynamic_extent;

  /// @brief The Span type.
  template<typename T, std::size_t extent = dynamicExtent>
  using Span = afft::thirdparty::span::span<T, extent>;
#endif

  /**
   * @brief Non-owning const view of a contiguous sequence of objects.
   * @tparam T Type of the elements.
   * @tparam extent Number of elements in the span.
   */
  template<typename T, std::size_t extent = dynamicExtent>
  using View = Span<const T, extent>;
} // namespace afft

#endif /* AFFT_SPAN_HPP */
