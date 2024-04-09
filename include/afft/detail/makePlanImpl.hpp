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

#ifndef AFFT_DETAIL_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_MAKE_PLAN_IMPL_HPP

#include <memory>
#include <utility>

#include "common.hpp"
#include "PlanImpl.hpp"
#include "gpu/makePlanImpl.hpp"
#include "cpu/makePlanImpl.hpp"

namespace afft::detail
{
  /**
   * @brief Make a plan implementation.
   * @tparam target Target.
   * @tparam Args Argument types.
   * @param args Arguments.
   * @return Plan implementation.
   */
  template<Target target, typename... Args>
  std::shared_ptr<PlanImpl> makePlanImpl(Args&&... args)
  {
    if constexpr (target == Target::cpu)
    {
      return cpu::makePlanImpl(std::forward<Args>(args)...);
    }
    else
    {
      return gpu::makePlanImpl(std::forward<Args>(args)...);
    }
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_IMPL_HPP */
