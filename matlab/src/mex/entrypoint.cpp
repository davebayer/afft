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

#include <afft/afft.hpp>
#include <matlabw/mex/mex.hpp>
#include <matlabw/mex/Function.hpp>

#include "packageManagement.hpp"
#include "plan.hpp"
#include "planCache.hpp"
#include "transform.hpp"

using namespace matlabw;

/// @brief Enumeration of all available calls.
enum class Call : std::uint32_t
{
  // Package management calls
  mlock = 0,
  munlock,
  clearPlanCache,

  // Plan calls
  planCreate = 1000,
  planExecute,
  planGetTransformParameters,
  planGetTargetParameters,

  // Forward transform calls
  fft = 2000,
  fft2,
  fftn,

  // Inverse transform calls
  ifft = 3000,
  ifft2,
  ifftn,
};

/**
 * @brief afft-matlab module entry point. This function is called by MATLAB.
 * @param lhs The left-hand side arguments.
 * @param rhs The right-hand side arguments. The first argument is the call type.
 */
void mex::Function::operator()(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  // When the library is first loaded, lock the package.
  static bool _initialLocker = [&](){ lock(); return true; }();

  // Initialize the afft library.
  afft::init();

  // Check the call type argument.
  if (!rhs[0].isScalar() || !rhs[0].isUint32())
  {
    throw mx::Exception{"afft:internal:invalidCallArgument", "invalid call argument type"};
  }

  mx::View<mx::ArrayCref> rhsSubspan{rhs.subspan(1)};

  // Dispatch the call.
  switch (static_cast<Call>(mx::NumericArrayCref<std::uint32_t>{rhs[0]}[0]))
  {
  // Package management calls
  case Call::mlock:
    lock();
    break;
  case Call::munlock:
    unlock();
    break;
  case Call::clearPlanCache:
    clearPlanCache(lhs, rhsSubspan);
    break;

  // Plan calls
  case Call::planCreate:
    planCreate(lhs, rhsSubspan);
    break;
  case Call::planExecute:
    planExecute(lhs, rhsSubspan);
    break;
  case Call::planGetTransformParameters:
    planGetTransformParameters(lhs, rhsSubspan);
    break;
  case Call::planGetTargetParameters:
    planGetTargetParameters(lhs, rhsSubspan);
    break;

  // Forward transform calls
  case Call::fft:
    fft(lhs, rhsSubspan);
    break;
  case Call::fft2:
    fft2(lhs, rhsSubspan);
    break;
  case Call::fftn:
    fftn(lhs, rhsSubspan);
    break;

  // Inverse transform calls
  case Call::ifft:
    ifft(lhs, rhsSubspan);
    break;
  case Call::ifft2:
    ifft2(lhs, rhsSubspan);
    break;
  case Call::ifftn:
    ifftn(lhs, rhsSubspan);
    break;
  default:
    throw mx::Exception{"afft:internal:invalidCall", "invalid call"};
  }
}
