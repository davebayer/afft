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

#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

template<std::size_t shapeRank>
struct Convert<afft::MemoryBlock<shapeRank>>
  : StructConvertBase<afft::MemoryBlock<shapeRank>, afft_MemoryBlock>
{
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    CxxType cxxValue{};
    cxxValue.starts  = afft::View<std::size_t>{cValue.starts, shapeRank};
    cxxValue.strides = afft::View<std::size_t>{cValue.strides, shapeRank};
    cxxValue.shape   = afft::View<std::size_t>{cValue.shape, shapeRank};
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue, std::size_t shapeRank)
  {
    CType cValue{};
    cValue.starts  = cxxValue.starts.data();
    cValue.strides = cxxValue.strides.data();
    cValue.shape   = cxxValue.shape.data();
  }
};

#endif /* ARCHITECTURE_HPP */
