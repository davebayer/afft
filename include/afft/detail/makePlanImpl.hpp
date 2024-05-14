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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "PlanImpl.hpp"
#include "cpu/makePlanImpl.hpp"
#if AFFT_GPU_IS_ENABLED
# include "gpu/makePlanImpl.hpp"
#endif

namespace afft::detail
{
  /**
   * @brief Create a PlanImpl object based on the given configuration.
   * @tparam ConfigT The configuration type.
   * @param config The configuration to use.
   * @param backendSelectParams The parameters for the transform backend selection.
   * @return std::unique_ptr<PlanImpl> The created PlanImpl object.
   */
  template<typename BackendSelectParametersT>
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makePlanImpl(const Config& config, const BackendSelectParametersT& backendSelectParams)
  {
    constexpr auto target = backendSelectParametersTarget<BackendSelectParametersT>;

    std::unique_ptr<PlanImpl> planImpl{};

    if (config.getTarget() != target)
    {
      throw makeException<std::runtime_error>("Invalid target");
    }

    if constexpr (target == Target::cpu)
    {
      planImpl = cpu::makePlanImpl(config, backendSelectParams);
    }
    else if constexpr (target == Target::gpu)
    {
#   if AFFT_GPU_IS_ENABLED
      planImpl = gpu::makePlanImpl(config, backendSelectParams);
#   else
      throw makeException<std::runtime_error>("GPU support is disabled");
#   endif
    }

    if (!planImpl)
    {
      throw makeException<std::runtime_error>("Failed to create PlanImpl object");
    }

    return planImpl;
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_IMPL_HPP */
