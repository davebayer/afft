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

#ifndef HELPERS_CFORMAT_HPP
#define HELPERS_CFORMAT_HPP

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace helpers
{
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
} // namespace helpers

#endif /* HELPERS_CFORMAT_HPP */
