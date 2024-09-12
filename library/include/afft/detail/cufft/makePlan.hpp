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

#ifndef AFFT_DETAIL_CUFFT_MAKE_PLAN_HPP
#define AFFT_DETAIL_CUFFT_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

// #include "mpst.hpp"
// #include "spmt.hpp"
#include "sp.hpp"

namespace afft::detail::cufft
{
  /**
   * @brief Create a plan for cuFFT.
   * @param desc The descriptor of the plan.
   * @param backendParams The backend parameters.
   * @return The plan or an error message.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const BackendParamsT& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    if (descImpl.getTransformHowManyRank() > 1)
    {
      throw Exception{Error::cufft, "omitting more than one dimension is not supported"};
    }

    if (descImpl.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw Exception{Error::cufft, "only interleaved complex format is supported"};
    }

    if (descImpl.getTransform() == Transform::dft)
    {
      const auto& dftDesc = descImpl.getTransformDesc<Transform::dft>();

      if (dftDesc.type == dft::Type::complexToReal &&
          descImpl.getPlacement() == Placement::outOfPlace &&
          !backendParams.allowDestructive)
      {
        throw Exception{Error::cufft, "preserving the source for complex-to-real transforms is not supported"};
      }
    }
    else
    {
      throw Exception{Error::cufft, "only DFT transforms are supported"};
    }

    if (const auto& prec = descImpl.getPrecision(); prec.execution != prec.source || prec.execution != prec.destination)
    {
      throw Exception{Error::cufft, "execution, source and destination must precision match"};
    }

    if (descImpl.getNormalization() != Normalization::none)
    {
      throw Exception{Error::cufft, "normalization is not supported"};
    }

    if constexpr (BackendParamsT::target == Target::cuda)
    {
      if constexpr (BackendParamsT::mpBackend == MpBackend::none)
      {
        return sp::makePlan(desc, backendParams);
      }
      else
      {
        throw Exception{Error::cufft, "only none backend is supported"};
      }
    }
    else
    {
      throw Exception{Error::cufft, "only CUDA target is supported"};
    }
  }
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_MAKE_PLAN_HPP */
