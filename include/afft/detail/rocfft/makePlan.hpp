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

#ifndef AFFT_DETAIL_ROCFFT_MAKE_PLAN_HPP
#define AFFT_DETAIL_ROCFFT_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"
#include "spst.hpp"
#include "spmt.hpp"

namespace afft::detail::rocfft
{
  /**
   * @brief Create a plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan([[maybe_unused]] const Desc& desc, const BackendParamsT&)
  {
    if (const auto tRank = desc.getTransformRank(); tRank > 3 || tRank == 0)
    {
      throw BackendError{Backend::rocfft, "only 1D, 2D and 3D transforms are supported"};
    }

    if (const auto hmRank = desc.getTransformHowManyRank(); hmRank > 1)
    {
      throw BackendError{Backend::rocfft, "only single and batched transforms are supported"};
    }

    if (desc.getTransform() != Transform::dft)
    {
      throw BackendError{Backend::rocfft, "only dft transform is supported"};
    }

    if (!desc.hasUniformPrecision())
    {
      throw BackendError{Backend::rocfft, "execution, source and destination precision must match"};
    }
    
    if constexpr (BackendParamsT::target == Target::gpu)
    {
#   if AFFT_GPU_IS_ENABLED
      if constexpr (BackendParamsT::distribution == Distribution::spst)
      {
        return spst::gpu::makePlan(desc);
      }
      else if constexpr (BackendParamsT::distribution == Distribution::spmt)
      {
        return spmt::gpu::makePlan(desc);
      }
      else
      {
        throw BackendError{Backend::rocfft, "only spst and spmt distributions are supported"};
      }
#   else
      throw BackendError{Backend::rocfft, "gpu support is disabled"};
#   endif
    }
    else
    {
      throw BackendError{Backend::rocfft, "only gpu target is supported"};
    }
  }
} // namespace afft::detail::rocfft

#endif // AFFT_DETAIL_ROCFFT_MAKE_PLAN_HPP
