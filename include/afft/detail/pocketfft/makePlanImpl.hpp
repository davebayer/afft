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

#ifndef AFFT_DETAIL_POCKETFFT_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_POCKETFFT_MAKE_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "PlanImpl.hpp"
#include "spst.hpp"

namespace afft::detail::pocketfft
{
  /**
   * @brief Create a plan implementation.
   * @tparam BackendParametersT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParametersT>
  [[nodiscard]] std::unique_ptr<PlanImpl>
  makePlanImpl(const Desc& desc, const BackendParametersT& backendParams)
  {
    if (desc.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw BackendError{Backend::pocketfft, "only interleaved complex format is supported"};
    }
    
    if constexpr (backendParams.target == Target::cpu)
    {
      if constexpr (backendParams.distribution == Distribution::spst)
      {
        return spst::cpu::makePlanImpl(desc);
      }
      else
      {
        throw BackendError{Backend::pocketfft, "only spst distribution is supported"};
      }
    }
    else
    {
      throw BackendError{Backend::pocketfft, "only cpu target is supported"};
    }
  }
} // namespace afft::detail::pocketfft

#endif /* AFFT_DETAIL_POCKETFFT_MAKE_PLAN_IMPL_HPP */

