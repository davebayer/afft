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

#ifndef AFFT_DETAIL_CPU_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_CPU_MAKE_PLAN_IMPL_HPP

#include <algorithm>
#include <memory>
#include <span>

#include "../../cpu.hpp"

// Include transform backend-specific plan implementations
#if AFFT_CPU_BACKEND_IS_ENABLED(FFTW3)
# include "fftw3/PlanImpl.hpp"
#endif
#if AFFT_CPU_BACKEND_IS_ENABLED(MKL)
# include "mkl/PlanImpl.hpp"
#endif
#if AFFT_CPU_BACKEND_IS_ENABLED(POCKETFFT)
# include "pocketfft/PlanImpl.hpp"
#endif

#include "../common.hpp"
#include "../cxx.hpp"
#include "../PlanImpl.hpp"

namespace afft::detail::cpu
{
  using namespace afft::cpu;

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
#   if AFFT_CPU_BACKEND_IS_ENABLED(FFTW3)
    case Backend::fftw3:
      return fftw3::makePlanImpl(config);
#   endif
#   if AFFT_CPU_BACKEND_IS_ENABLED(MKL)
    case Backend::mkl:
      return mkl::makePlanImpl(config);
#   endif
#   if AFFT_CPU_BACKEND_IS_ENABLED(POCKETFFT)
    case Backend::pocketfft:
      return pocketfft::makePlanImpl(config);
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
  makePlanImpl(const Config& config, const afft::cpu::BackendSelectParameters& backendSelectParams)
  {
    Span<const Backend> backends = (!backendSelectParams.backends.empty())
                                      ? backendSelectParams.backends : afft::cpu::defaultBackendList;

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
} // namespace afft::detail::cpu

#endif /* AFFT_DETAIL_CPU_MAKE_PLAN_IMPL_HPP */
