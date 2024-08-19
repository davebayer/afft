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
#include <mex/mex.hpp>
#include <mex/Function.hpp>

/// @brief Enumeration of all available calls.
enum class Call : std::uint32_t
{
  // Package management calls
  clearCache  = 0,

  // Plan calls
  planCreate  = 1000,
  planExecute,

  // Forward transform calls
  fft         = 2000,
  fft2,
  fftn,

  // Inverse transform calls
  ifft        = 3000,
  ifft2,
  ifftn,
};

/// @brief Plan cache. Used to store all created plans.
afft::PlanCache planCache{};

/**
 * @brief afft-matlab module entry point. This function is called by MATLAB.
 * @param lhs The left-hand side arguments.
 * @param rhs The right-hand side arguments. The first argument is the call type.
 */
void mex::Function::operator()(mex::Span<mex::Array> lhs, mex::View<mex::ArrayCref> rhs)
{
  // Keep the function in the memory even when called `clear all`. Prevents MATLAB crashes due to invalid pointers.
  if (!isLocked())
  {
    lock();
  }

  // Initialize the afft library.
  afft::init();

  // Check the call type argument.
  if (!rhs[0].isScalar() || !rhs[0].isUint32())
  {
    throw mex::Exception{"afft:internal:invalidCallArgument", "invalid call argument type"};
  }

  // Dispatch the call.
  switch (static_cast<Call>(mex::NumericArrayCref<std::uint32_t>{rhs[0]}[0]))
  {
  case Call::planCreate:
    planCreate(lhs, rhs.subspan(1));
    break;
  case Call::planExecute:
    break;
  case Call::fft:
    break;
  case Call::fft2:
    break;
  case Call::fftn:
    break;
  case Call::ifft:
    break;
  case Call::ifft2:
    break;
  case Call::ifftn:
    break;
  default:
    throw mex::Exception{"afft:internal:invalidCall", "invalid call"};
  }
}
