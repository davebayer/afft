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

#ifndef AFFT_DETAIL_CLFFT_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_CLFFT_MAKE_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "spst.hpp"

namespace afft::detail::clfft
{
  /**
   * @brief Create a plan implementation for clFFT.
   * @param desc The descriptor of the plan.
   * @param initParams The initialization parameters.
   * @return The plan implementation or an error message.
   */
  [[nodiscard]] inline cxx::expected<std::unique_ptr<detail::PlanImpl>, std::string>
  makePlanImpl(const Desc& desc, const InitParams& initParams)
  {
    if (desc.getTransformRank() == 0 || desc.getTransformRank() > 3)
    {
      return cxx::unexpected("clFFT only supports 1D, 2D, and 3D transforms");
    }

    if (desc.getTransformHowManyRank() > 1)
    {
      return cxx::unexpected("clFFT does not support omiting multiple dimensions");
    }

    if (desc.getTransform() == Transform::dft)
    {
      const auto& dftDesc = desc.getTransformDesc<Transform::dft>();

      if ((dftDesc.type == dft::Type::complexToReal || dftDesc == dft::Type::realToComplex)
          && desc.getComplexFormat() == ComplexFormat::planar)
      {
        return cxx::unexpected("clFFT only supports interleaved complex format for complex-to-real and real-to-complex transforms");
      }
    }
    else
    {
      return cxx::unexpected("cuFFT only supports DFT transforms");
    }

    if (const auto& prec = desc.getPrecision(); prec.execution == prec.source && prec.execution == prec.destination)
    {
      if (prec.execution != Precision::single && prec.execution != Precision::double)
      {
        return cxx::unexpected("clFFT only supports single and double precision");
      }
    }
    else
    {
      return cxx::unexpected("clFFT only supports same precision for execution, source and destination");
    }

    switch (desc.getTarget())
    {
    case Target::gpu:
      switch (desc.getDistribution())
      {
      case Distribution::spst:
        return spst::gpu::PlanImpl::make(desc, initParams.initEffort);
      default:
        return cxx::unexpected("Unsupported distribution");
      }
    default:
      return cxx::unexpected("Unsupported target");
    }
  }
} // namespace afft::detail::clfft

#endif /* AFFT_DETAIL_CLFFT_MAKE_PLAN_IMPL_HPP */
