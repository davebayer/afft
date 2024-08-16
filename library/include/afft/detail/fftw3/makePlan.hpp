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

#ifndef AFFT_DETAIL_FFTW3_MAKE_PLAN_HPP
#define AFFT_DETAIL_FFTW3_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"
#include "sp.hpp"

namespace afft::detail::fftw3
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
  makePlan(const Desc& desc, const BackendParamsT& backendParams)
  {
    if (desc.getTargetCount() != 1)
    {
      throw Exception{Error::fftw3, "only single target is supported"};
    }

    switch (desc.getTransform())
    {
    case Transform::dft:
    case Transform::dtt:
      break;
    default:
      throw Exception{Error::fftw3, "unsupported transform"};
    }

    if (!desc.hasUniformPrecision())
    {
      throw Exception{Error::fftw3, "only uniform precision is supported"};
    }
    
    if constexpr (BackendParamsT::mpBackend == MpBackend::none)
    {
      if constexpr (BackendParamsT::target == Target::cpu)
      {
        return sp::cpu::makePlan(desc, backendParams);
      }
      else
      {
        throw Exception{Error::fftw3, "only cpu target is supported"};
      }
    }
    else if constexpr (BackendParamsT::mpBackend == MpBackend::mpi)
    {
      throw Exception{Error::fftw3, "distributed computation over MPI is not supported yet"};
    }
    else
    {
      throw Exception{Error::fftw3, "unsupported mp backend"};
    }
  }
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_MAKE_PLAN_HPP */

