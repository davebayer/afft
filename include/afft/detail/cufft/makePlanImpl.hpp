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

#ifndef AFFT_DETAIL_CUFFT_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_CUFFT_MAKE_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "spst.hpp"
#include "spmt.hpp"
#include "mpst.hpp"

namespace afft::detail::cufft
{
  /**
   * @brief Create a plan implementation for cuFFT.
   * @param desc The descriptor of the plan.
   * @param initParams The initialization parameters.
   * @return The plan implementation or an error message.
   */
  [[nodiscard]] inline cxx::expected<std::unique_ptr<detail::PlanImpl>, std::string>
  makePlanImpl(const Desc& desc, const InitParams& initParams)
  {
    if (desc.getTransformRank() == 0 || desc.getTransformRank() > 3)
    {
      return cxx::unexpected("cuFFT only supports 1D, 2D, and 3D transforms");
    }

    if (desc.getTransformHowManyRank() > 1)
    {
      return cxx::unexpected("cuFFT does not support omiting multiple dimensions");
    }

    if (desc.getComplexFormat() != ComplexFormat::interleaved)
    {
      return cxx::unexpected("cuFFT only supports interleaved complex format");
    }

    if (desc.getTransform() == Transform::dft)
    {
      const auto& dftDesc = desc.getTransformDesc<Transform::dft>();

      if (dftDesc.type == dft::Type::complexToReal && desc.getPreserveSource())
      {
        return cxx::unexpected("cuFFT does not support preserving the source for complex-to-real transforms");
      }
    }
    else
    {
      return cxx::unexpected("cuFFT only supports DFT transforms");
    }

    if (const auto& prec = desc.getPrecision(); prec.execution != prec.source || prec.execution != prec.destination)
    {
      return cxx::unexpected("cuFFT only supports the same precision for execution, source and destination");
    }

    if (desc.getNormalize() != Normalize::none)
    {
      return cxx::unexpected("cuFFT does not support normalization");
    }

    switch (desc.getTarget())
    {
    case Target::gpu:
      switch (desc.getDistribution())
      {
      case Distribution::spst:
        return spst::gpu::PlanImpl::make(desc, initParams.initEffort);
      case Distribution::spmt:
        return spmt::gpu::PlanImpl::make(desc, initParams.initEffort);
      case Distribution::mpst:
        return mpst::gpu::PlanImpl::make(desc, initParams.initEffort);
      default:
        return cxx::unexpected("Unsupported distribution");
      }
      break;
    default:
      return cxx::unexpected("Unsupported target");
    }
  }
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_MAKE_PLAN_IMPL_HPP */
