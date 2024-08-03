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

#include <mex.h>
#include <afft/afft.hpp>

#if TARGET_API_VERSION != 800
# error "This version of the afft library requires the MATLAB R2018a API."
#endif

extern "C" void
mexFunction(int            nlhs,
            mxArray*       plhs[],
            int            nrhs,
            const mxArray* prhs[])
try
{
  if (nlhs != 1)
  {
    mexErrMsgIdAndTxt("afft:fft:nlhs", "One output required.");
  }

  if (nrhs != 1 && nrhs != 2)
  {
    mexErrMsgIdAndTxt("afft:fft:nrhs", "One or two inputs required.");
  }
  
  mxArray* anySrc = const_cast<mxArray*>(prhs[0]);

  const std::size_t  ndims = mxGetNumberOfDimensions(anySrc);
  const std::size_t* dims  = mxGetDimensions(anySrc);

  if (ndims > afft::maxDimCount)
  {
    mexErrMsgIdAndTxt("afft:fft:ndims", "Too many dimensions.");
  }

  mxArray* src{};

  if (mexCallMATLAB(1, &src, 1, &anySrc, "double"))
  {
    mexErrMsgIdAndTxt("afft:fft:conversion", "Failed to convert input to double.");
  }
  if (!mxMakeArrayComplex(src))
  {
    mexErrMsgIdAndTxt("afft:fft:complex", "Failed to make input complex.");
  }

  mxArray* dst = mxCreateUninitNumericArray(ndims, const_cast<std::size_t*>(dims), mxDOUBLE_CLASS, mxCOMPLEX);

  if (dst == nullptr)
  {
    mexErrMsgIdAndTxt("afft:fft:allocation", "Failed to allocate output array.");
  }

  std::array<afft::Size, afft::maxDimCount> shape{};
  std::transform(dims,
                 dims + ndims,
                 shape.rbegin() + (shape.size() - ndims),
                 [](const std::size_t dim) { return static_cast<afft::Size>(dim); });

  afft::Axis axis = static_cast<afft::Axis>(ndims - 1);

  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::forward;
  dftParams.precision     = afft::makePrecision<double>();
  dftParams.shape         = afft::View<afft::Size>{shape.data(), ndims};
  dftParams.axes          = afft::makeScalarView(axis);
  dftParams.normalization = afft::Normalization::none;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.destructive   = true;
  dftParams.type          = afft::dft::Type::complexToComplex;

  afft::cpu::Parameters cpuParams{};
  cpuParams.threadLimit = 4;

  auto plan = afft::makePlan(dftParams, cpuParams);

  plan->executeUnsafe(mxGetData(src), mxGetData(dst));

  plhs[0] = dst;
}
catch (...)
{
  mexErrMsgIdAndTxt("afft:fft:exception", "An exception occurred.");
}
