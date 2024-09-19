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

#include "toolboxManagement.hpp"
#include "plan.hpp"
#include "planCache.hpp"
#include "transform.hpp"

using namespace matlabw;

/// @brief Enumeration of all available calls.
enum class Call : std::uint32_t
{
  // Toolbox management calls
  clearPlanCache = 0,

  // Plan calls
  planCreate = 1000,
  planExecute,
  planGetTransformParameters,
  planGetTargetParameters,

  // Discrete Fourier transform calls
  fft = 2000,
  fft2,
  fftn,
  ifft,
  ifft2,
  ifftn,
  rfft,
  rfft2,
  rfftn,
  irfft,
  irfft2,
  irfftn,

  // Discrete Hartley transform calls
  dht = 3000,
  dht2,
  dhtn,
  idht,
  idht2,
  idhtn,

  // Discrete cosine transform calls
  dct = 4000,
  dct2,
  dctn,
  idct,
  idct2,
  idctn,

  // Discrete sine transform calls
  dst = 5000,
  dst2,
  dstn,
  idst,
  idst2,
  idstn,

  // Discrete trigonometric transform calls
  dtt = 6000,
  dtt2,
  dttn,
  idtt,
  idtt2,
  idttn,
};

/**
 * @brief afft-matlab module entry point. This function is called by MATLAB.
 * @param lhs The left-hand side arguments.
 * @param rhs The right-hand side arguments. The first argument is the call type.
 */
void mex::Function::operator()(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  // When the library is first loaded, lock the mex.
  if (!isLocked())
  {
    lock();
  }

  // Initialize the afft library.
  afft::init();

  // Check the call type argument.
  if (!rhs[0].isScalar() || !rhs[0].isUint32())
  {
    throw mx::Exception{"afft:internal:invalidCallArgument", "invalid call argument type"};
  }

  mx::View<mx::ArrayCref> rhsSubspan{rhs.subspan(1)};

  // Dispatch the call.
  switch (static_cast<Call>(rhs[0].getScalarAs<std::uint32_t>()))
  {
  // Toolbox management calls
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

  // Discrete Fourier transform calls
  case Call::fft:
    fft(lhs, rhsSubspan);
    break;
  case Call::fft2:
    fft2(lhs, rhsSubspan);
    break;
  case Call::fftn:
    fftn(lhs, rhsSubspan);
    break;
  case Call::ifft:
    ifft(lhs, rhsSubspan);
    break;
  case Call::ifft2:
    ifft2(lhs, rhsSubspan);
    break;
  case Call::ifftn:
    ifftn(lhs, rhsSubspan);
    break;

  // Real-data discrete Fourier transform calls
  case Call::rfft:
    rfft(lhs, rhsSubspan);
    break;
  case Call::rfft2:
    rfft2(lhs, rhsSubspan);
    break;
  case Call::rfftn:
    rfftn(lhs, rhsSubspan);
    break;
  case Call::irfft:
    irfft(lhs, rhsSubspan);
    break;
  case Call::irfft2:
    irfft2(lhs, rhsSubspan);
    break;
  case Call::irfftn:
    irfftn(lhs, rhsSubspan);
    break;

  // Discrete Hartley transform calls
  // case Call::dht:
  //   dht(lhs, rhsSubspan);
  //   break;
  // case Call::dht2:
  //   dht2(lhs, rhsSubspan);
  //   break;
  // case Call::dhtn:
  //   dhtn(lhs, rhsSubspan);
  //   break;
  // case Call::idht:
  //   idht(lhs, rhsSubspan);
  //   break;
  // case Call::idht2:
  //   idht2(lhs, rhsSubspan);
  //   break;
  // case Call::idhtn:
  //   idhtn(lhs, rhsSubspan);
  //   break;

  // Discrete cosine transform calls
  case Call::dct:
    dct(lhs, rhsSubspan);
    break;
  case Call::dct2:
    dct2(lhs, rhsSubspan);
    break;
  case Call::dctn:
    dctn(lhs, rhsSubspan);
    break;
  case Call::idct:
    idct(lhs, rhsSubspan);
    break;
  case Call::idct2:
    idct2(lhs, rhsSubspan);
    break;
  case Call::idctn:
    idctn(lhs, rhsSubspan);
    break;

  // Discrete sine transform calls
  case Call::dst:
    dst(lhs, rhsSubspan);
    break;
  case Call::dst2:
    dst2(lhs, rhsSubspan);
    break;
  case Call::dstn:
    dstn(lhs, rhsSubspan);
    break;
  // case Call::idst:
  //   idst(lhs, rhsSubspan);
  //   break;
  // case Call::idst2:
  //   idst2(lhs, rhsSubspan);
  //   break;
  // case Call::idstn:
  //   idstn(lhs, rhsSubspan);
  //   break;

  // Discrete trigonometric transform calls
  // case Call::dtt:
  //   dtt(lhs, rhsSubspan);
  //   break;
  // case Call::dtt2:
  //   dtt2(lhs, rhsSubspan);
  //   break;
  // case Call::dttn:
  //   dttn(lhs, rhsSubspan);
  //   break;
  // case Call::idtt:
  //   idtt(lhs, rhsSubspan);
  //   break;
  // case Call::idtt2:
  //   idtt2(lhs, rhsSubspan);
  //   break;
  // case Call::idttn:
  //   idttn(lhs, rhsSubspan);
  //   break;
  
  default:
    throw mx::Exception{"afft:internal:invalidCall", "invalid call"};
  }
}
