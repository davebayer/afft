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

#include <mex/mex.hpp>
#include <mex/Function.hpp>
#include <afft/afft.hpp>

void mex::Function::operator()(mex::Span<mex::Array> lhs, mex::View<mex::ArrayCref> rhs)
try
{
  if (lhs.size() != 1)
  {
    throw mex::Exception{"afft:fft:nlhs", "One output required."};
  }

  if (rhs.size() != 1 && rhs.size() != 2)
  {
    throw mex::Exception{"afft:fft:nrhs", "One or two inputs required."};
  }

  const auto ndims = rhs[0].getRank();
  const auto dims  = rhs[0].getDims();

  if (ndims > afft::maxDimCount)
  {
    throw mex::Exception{"afft:fft:ndims", "Too many dimensions."};
  }

  mex::Array src{};

  mex::call(mex::makeScalarSpan(src), {{rhs[0]}}, "double");

  if (!mxMakeArrayComplex(src.get()))
  {
    throw mex::Exception{"afft:fft:complex", "Failed to make input complex."};
  }

  auto dst = mex::makeUninitNumericArray<std::complex<double>>(dims);

  std::array<afft::Size, afft::maxDimCount> shape{};
  std::transform(dims.begin(),
                 dims.begin() + ndims,
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

  plan->executeUnsafe(src.getData(), dst.getData());

  lhs[0] = std::move(dst);
}
catch (const afft::Exception& e)
{
  throw mex::Exception{"afft::fft::afftException", e.what()};
}
