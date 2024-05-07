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

#ifndef AFFT_DETAIL_GPU_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_GPU_MAKE_PLAN_IMPL_HPP

#include <algorithm>
#include <memory>
#include <span>

#include "../../gpu.hpp"

#if AFFT_GPU_BACKEND_IS_ENABLED(CLFFT)
# include "clfft/PlanImpl.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(CUFFT)
# include "cufft/PlanImpl.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(HIPFFT)
# include "hipfft/PlanImpl.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(ROCFFT)
# include "rocfft/PlanImpl.hpp"
#endif
#if AFFT_GPU_BACKEND_IS_ENABLED(VKFFT)
# include "vkfft/PlanImpl.hpp"
#endif

#include "../common.hpp"
#include "../cxx.hpp"
#include "../PlanImpl.hpp"

namespace afft::detail::gpu
{
  using namespace afft::gpu;

  /**
   * @brief Create a plan implementation for the specified backend.
   * @param config The configuration.
   * @param backend The backend.
   * @return The plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makePlanImpl([[maybe_unused]] const Config& config, Backend backend)
  try
  {
    switch (backend)
    {
#   if AFFT_GPU_BACKEND_IS_ENABLED(CLFFT)
    case Backend::clfft:
      return clfft::makePlanImpl(config);
#   endif
#   if AFFT_GPU_BACKEND_IS_ENABLED(CUFFT)
    case Backend::cufft:
      return cufft::makePlanImpl(config);
#   endif
#   if AFFT_GPU_BACKEND_IS_ENABLED(HIPFFT)
    case Backend::hipfft:
      return hipfft::makePlanImpl(config);
#   endif
#   if AFFT_GPU_BACKEND_IS_ENABLED(ROCFFT)
    case Backend::rocfft:
      return rocfft::makePlanImpl(config);
#   endif
#   if AFFT_GPU_BACKEND_IS_ENABLED(VKFFT)
    case Backend::vkfft:
      return vkfft::makePlanImpl(config);
#   endif
    default:
      return nullptr;
    }
  }
  catch (...)
  {
    return nullptr;
  }

  /**
   * @brief Create a plan implementation for the first available backend.
   * @param config The configuration.
   * @param backends The backends to try.
   * @return The plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makeFirstPlanImpl(const Config& config, Span<const Backend> backends)
  {
    std::unique_ptr<PlanImpl> planImpl{};

    for (const auto backend : backends)
    {
      if ((planImpl = makePlanImpl(config, backend)))
      {
        break;
      }
    }

    return planImpl;
  }

  /**
   * @brief Create the best plan implementation.
   * @param config The configuration.
   * @param backends The backends to try.
   * @return The plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makeBestPlanImpl(const Config& config, Span<const Backend> backends)
  {
    // struct Item
    // {
    //   std::unique_ptr<PlanImpl>     planImpl{};
    //   std::chrono::duration<double> time{std::chrono::duration<double>::max()};
    // };

    // std::array<Item, backendCount> planImpls{};

    // for (const auto backend : backends)
    // {
    //   std::size_t i = static_cast<std::size_t>(backend);

    //   planImpls[i].planImpl = makePlanImpl(config, backend);
    // }

    // // TODO allocate memory and perform measurements, then choose the best plan

    // auto best = std::min_element(backends.begin(),
    //                              backends.end(),
    //                              [&](const Backend a, const Backend b)
    // {
    //   return planImpls[static_cast<std::size_t>(a)].time < planImpls[static_cast<std::size_t>(b)].time;
    // });

    // return std::move(planImpls[static_cast<std::size_t>(*best)].planImpl);
    throw makeException<std::logic_error>("Not implemented");
  }

  /**
   * @brief Create a plan implementation.
   * @param config The configuration.
   * @param backendSelectParams The backend selection parameters.
   * @return The plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makePlanImpl(const Config& config, const afft::gpu::BackendSelectParameters& backendSelectParams)
  {
    Span<const Backend> backends = (!backendSelectParams.backends.empty())
                                      ? backendSelectParams.backends : afft::gpu::defaultBackendList;

    switch (backendSelectParams.strategy)
    {
    case BackendSelectStrategy::first:
      return makeFirstPlanImpl(config, backends);
    case BackendSelectStrategy::best:
      return makeBestPlanImpl(config, backends);
    default:
      cxx::unreachable();
    }
  }
} // namespace afft::detail::gpu

#endif /* AFFT_DETAIL_GPU_MAKE_PLAN_IMPL_HPP */
