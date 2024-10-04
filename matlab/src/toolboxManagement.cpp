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
#include "parser.hpp"
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
 * @brief Check if the backend has GPU support.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 0.
 */
void hasGpuSupport(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 0)
  {
    throw mx::Exception{"afft:hasGpuSupport:invalidArgumentCount", "invalid argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:hasGpuSupport:invalidOutputCount", "invalid output count"};
  }

#ifdef MATLABW_ENABLE_GPU
  lhs[0] = mx::makeLogicalScalar(true);
#else
  lhs[0] = mx::makeLogicalScalar(false);
#endif
}

/**
 * @brief Check if the backend is available.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 1.
 */
void hasBackend(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 1)
  {
    throw mx::Exception{"afft:hasBackend:invalidArgumentCount", "invalid argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:hasBackend:invalidOutputCount", "invalid output count"};
  }

  BackendParser backendParser{};

  switch (backendParser(rhs[0]))
  {
  case afft::Backend::cufft:
# ifdef AFFT_ENABLE_CUFFT
    lhs[0] = mx::makeLogicalScalar(true);
# else
    lhs[0] = mx::makeLogicalScalar(false);
# endif
    break;
  case afft::Backend::fftw3:
# ifdef AFFT_ENABLE_FFTW3
    lhs[0] = mx::makeLogicalScalar(true);
# else
    lhs[0] = mx::makeLogicalScalar(false);
# endif
    break;
  case afft::Backend::mkl:
# ifdef AFFT_ENABLE_MKL
    lhs[0] = mx::makeLogicalScalar(true);
# else
    lhs[0] = mx::makeLogicalScalar(false);
# endif
    break;
  case afft::Backend::pocketfft:
# ifdef AFFT_ENABLE_POCKETFFT
    lhs[0] = mx::makeLogicalScalar(true);
# else
    lhs[0] = mx::makeLogicalScalar(false);
# endif
    break;
  case afft::Backend::vkfft:
# ifdef AFFT_ENABLE_VKFFT
    lhs[0] = mx::makeLogicalScalar(true);
# else
    lhs[0] = mx::makeLogicalScalar(false);
# endif
    break;
  default:
    throw mx::Exception{"afft:hasBackend:invalidBackend", "invalid backend"};
  }
}

/**
 * @brief Check if the cufft backend has callbacks.
 * @param lhs Left-hand side array of size 1.
 * @param rhs Right-hand side array of size 0.
 */
void hasCufftCallbacks(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 0)
  {
    throw mx::Exception{"afft:hasCufftCallbacks:invalidArgumentCount", "invalid argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:hasCufftCallbacks:invalidOutputCount", "invalid output count"};
  }

#if defined(AFFT_ENABLE_CUFFT) && CUFFT_VERSION >= 11300
  lhs[0] = mx::makeLogicalScalar(true);
#else
  lhs[0] = mx::makeLogicalScalar(false);
#endif
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
