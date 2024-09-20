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

#include "toolboxManagement.hpp"
#include "planCache.hpp"

using namespace matlabw;

/**
 * @brief Get the maximum number of dimensions supported by the backend.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 0.
 */
void maxDimCount(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 0)
  {
    throw mx::Exception{"afft:maxDimCount:invalidArgumentCount", "invalid argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:maxDimCount:invalidOutputCount", "invalid output count"};
  }

  lhs[0] = mx::makeNumericScalar<std::uint64_t>(static_cast<std::uint64_t>(afft::maxDimCount));
}

/**
 * @brief Clear the plan cache.
 * @param lhs Left-hand side array of size 0.
 * @param rhs Right-hand side array of size 0.
 */
void clearPlanCache(mx::Span<mx::Array>, mx::View<mx::ArrayCref>)
{
  // If the plan cache epoch overflows, throw an exception. This should probably never happen.
  if (planCacheEpoch == std::numeric_limits<std::uint64_t>::max())
  {
    throw mx::Exception{"afft:clearPlanCache:epochOverflow", "plan cache epoch overflow"};
  }

  planCache.clear();
  ++planCacheEpoch;
}
