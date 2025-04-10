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

#ifndef AFFT_DETAIL_MKL_MAKE_PLAN_HPP
#define AFFT_DETAIL_MKL_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

#include "sp.hpp"
#ifdef AFFT_ENABLE_MPI
# include "mpi.hpp"
#endif

namespace afft::detail::mkl
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
    if (desc.getTargetCount() != 1)
    {
      throw Exception{Error::mkl, "only single target is supported"};
    }
    
    if constexpr (BackendParamsT::mpBackend == MpBackend::none)
    {
      return sp::makePlan(desc, backendParams);
    }
    else if constexpr (BackendParamsT::mpBackend == MpBackend::mpi)
    {
#   ifdef AFFT_ENABLE_MPI
      return mpi::makePlan(desc, backendParams);
#   else
      throw Exception{Error::mkl, "mpi multi-process backend is not enabled"};
#   endif
    }
    else
    {
      throw Exception{Error::mkl, "unsupported mp backend"};
    }
  }
} // namespace afft::detail::mkl

#endif /* AFFT_DETAIL_MKL_MAKE_PLAN_HPP */

