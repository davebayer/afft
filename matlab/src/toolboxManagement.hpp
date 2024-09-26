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

#ifndef TOOLBOX_MANAGEMENT_HPP
#define TOOLBOX_MANAGEMENT_HPP

#include <afft/afft.hpp>
#include <matlabw/mx/mx.hpp>

/**
 * @brief Get the maximum number of dimensions supported by the backend.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 0.
 */
void maxDimCount(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Check if the backend has GPU support.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 0.
 */
void hasGpuSupport(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Clear the plan cache.
 * @param lhs Left-hand side array of size 0.
 * @param rhs Right-hand side array of size 0.
 */
void clearPlanCache(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

#endif /* TOOLBOX_MANAGEMENT_HPP */
