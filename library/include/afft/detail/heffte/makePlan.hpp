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

#ifndef AFFT_DETAIL_HEFFTE_MAKE_PLAN_HPP
#define AFFT_DETAIL_HEFFTE_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "Plan.hpp"
#include "mpst.hpp"

namespace afft::detail::heffte
{
  /**
   * @brief Create a plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<Plan>
  makePlan(const Desc& desc, const BackendParamsT& backendParams)
  {
    if (desc.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw BackendError{Backend::heffte, "only interleaved complex format is supported"};
    }

    if (!desc.hasUniformPrecision())
    {
      throw BackendError{Backend::heffte, "only same precision for execution, source and destination is supported"};
    }

    if (desc.getTransform() != Transform::dft)
    {
      throw BackendError{Backend::heffte, "only DFT transform is supported"};
    }

    if (desc.getPlacement() != Placement::outOfPlace)
    {
      throw BackendError{Backend::heffte, "only out-of-place placement is supported"};
    }

    if constexpr (backendParams.distribution == Distribution::mpst)
    {
      if constexpr (backendParams.target == Target::cpu)
      {
        return mpst::cpu::makePlan(desc, backendParams.heffte);
      }
      else if constexpr (backendParams.target == Target::gpu)
      {
#       if defined(AFFT_ENABLE_CUDA) || defined(AFFT_ENABLE_HIP)
        return mpst::gpu::makePlan(desc, backendParams.heffte);
#       else
        throw BackendError{Backend::heffte, "unsupported GPU backend, only CUDA and HIP are supported"};
#       endif
#     else
        throw BackendError{Backend::heffte, "gpu support is not enabled"};
#     endif
      }
      else
      {
        throw BackendError{Backend::heffte, "unsupported target"};
      }
    }
    else
    {
      throw BackendError{Backend::heffte, "only spst distribution is supported"};
    }
  }
} // namespace afft::detail::heffte

#endif /* AFFT_DETAIL_HEFFTE_MAKE_PLAN_HPP */

