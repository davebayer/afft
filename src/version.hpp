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

#ifndef VERSION_HPP
#define VERSION_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

// Version
template<>
struct Convert<afft::Version>
  : StructConvertBase<afft::Version, afft_Version>
{
  static_assert(std::is_same_v<decltype(afft::Version::major), decltype(afft_Version::major)>);
  static_assert(std::is_same_v<decltype(afft::Version::minor), decltype(afft_Version::minor)>);
  static_assert(std::is_same_v<decltype(afft::Version::patch), decltype(afft_Version::patch)>);

  /**
   * @brief Convert from C to C++.
   * @param cValue C value.
   * @return C++ value.
   */
  [[nodiscard]] static constexpr afft::Version fromC(const afft_Version& cValue) noexcept
  {
    return afft::Version{cValue.major, cValue.minor, cValue.patch};
  }

  /**
   * @brief Convert from C++ to C.
   * @param version C++ value.
   * @return C value.
   */
  [[nodiscard]] static constexpr afft_Version toC(const afft::Version& cxxValue) noexcept
  {
    return afft_Version{cxxValue.major, cxxValue.minor, cxxValue.patch};
  }
};

#endif /* VERSION_HPP */
