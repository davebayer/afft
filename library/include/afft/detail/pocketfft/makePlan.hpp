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

#ifndef AFFT_DETAIL_POCKETFFT_MAKE_PLAN_HPP
#define AFFT_DETAIL_POCKETFFT_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"
#include "sp.hpp"

namespace afft::detail::pocketfft
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
  makePlan(const Description& desc, const BackendParamsT& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    if (descImpl.getTargetCount() != 1)
    {
      throw Exception{Error::pocketfft, "only single target is supported"};
    }

    if (descImpl.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw Exception{Error::pocketfft, "only interleaved complex format is supported"};
    }
    
    if constexpr (BackendParamsT::mpBackend == MpBackend::none)
    {
      if constexpr (BackendParamsT::target == Target::cpu)
      {
        return sp::cpu::makePlan(desc, backendParams);
      }
      else
      {
        throw Exception{Error::pocketfft, "only cpu target is supported"};
      }
    }
    else
    {
      throw Exception{Error::pocketfft, "distribution over multiple processes is not supported"};
    }
  }
} // namespace afft::detail::pocketfft

#endif /* AFFT_DETAIL_POCKETFFT_MAKE_PLAN_HPP */

