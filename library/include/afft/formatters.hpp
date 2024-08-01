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

#ifndef AFFT_FORMATTERS_HPP
#define AFFT_FORMATTERS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "version.hpp"

/// @brief Formatter for afft::Version
AFFT_EXPORT template<>
struct std::formatter<afft::Version>
{
  /**
   * @brief Parse a format string.
   * @param ctx Format context
   * @return std::format_parse_context::iterator
   */
  [[nodiscard]] constexpr auto parse(std::format_parse_context& ctx) const noexcept
    -> std::format_parse_context::iterator
  {
    std::format_parse_context::iterator it{};

    for (it = ctx.begin(); it != ctx.end() && *it != '}'; ++it) {}

    return it;
  }

  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Version& version, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{}.{}.{}", version.major, version.minor, version.patch);
  }
};

#endif /* AFFT_FORMATTERS_HPP */
