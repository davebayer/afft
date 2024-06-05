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

#ifndef AFFT_DETAIL_MAKE_PLAN_HPP
#define AFFT_DETAIL_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "Desc.hpp"
#include "../Plan.hpp"
#include "../backend.hpp"

#if AFFT_BACKEND_IS_ENABLED(CLFFT)
# include "clfft/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(CUFFT)
# include "cufft/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(FFTW3)
# include "fftw3/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(MKL)
# include "mkl/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(POCKETFFT)
# include "pocketfft/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(ROCFFT)
# include "rocfft/makePlan.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(VKFFT)
# include "vkfft/makePlan.hpp"
#endif

namespace afft::detail
{
  struct DefaultBackendParams
  {
    template<typename BackendParamsT>
    [[nodiscard]] constexpr operator BackendParamsT() const
    {
      static_assert(isBackendParameters<BackendParamsT>, "invalid backend parameters type");

      return {};
    }
  };

  template<typename BackendParamsT>
  inline constexpr bool isKnownBackendParams = isBackendParameters<BackendParamsT> || std::is_same_v<BackendParamsT, DefaultBackendParams>;

  /**
   * @brief For each backend in the backend mask, call the function.
   * @tparam Fn Function type.
   * @param backendMask Backend mask.
   * @param fn Function to call for each backend.
   */
  template<typename Fn>
  auto forEachBackend(BackendMask backendMask, Fn&& fn)
    -> AFFT_RET_REQUIRES(void, AFFT_PARAM(std::is_invocable_v<Fn, Backend>))
  {
    for (BackendUnderlyingType i{}; i < cxx::to_underlying(Backend::_count); ++i)
    {
      const Backend backend = static_cast<Backend>(i);

      if ((backendMask & backend) != BackendMask::empty)
      {
        fn(backend);
      }
    }
  }

  /**
   * @brief For each backend in the backend mask, call the function.
   * @tparam Fn Function type.
   * @param backendMask Backend mask.
   * @param backendOrder Backend order.
   * @param fn Function to call for each backend.
   */
  template<typename Fn>
  auto forEachBackend(BackendMask backendMask, View<Backend> backendOrder, Fn&& fn)
    -> AFFT_RET_REQUIRES(void, AFFT_PARAM(std::is_invocable_v<Fn, Backend>))
  {
    for (const Backend backend : backendOrder)
    {
      if ((backendMask & backend) != BackendMask::empty)
      {
        fn(backend);
      }

      backendMask = backendMask & (~backend);
    }

    if (backendMask != BackendMask::empty)
    {
      forEachBackend(backendMask, std::forward<Fn>(fn));
    }
  }

  /**
   * @brief Get supported backend mask for the specified target and distribution.
   * @tparam target Target.
   * @tparam distrib Distribution.
   * @return Supported backend mask.
   */
  template<Target target, Distribution distrib>
  [[nodiscard]] constexpr BackendMask getSupportedBackendMask()
  {
    static_assert(isValid(target), "Invalid target");
    static_assert(isValid(distrib), "Invalid distribution");

    if constexpr (target == Target::cpu)
    {
      if constexpr (distrib == Distribution::spst)
      {
        return spst::cpu::supportedBackendMask;
      }
      else if constexpr (distrib == Distribution::mpst)
      {
        return mpst::cpu::supportedBackendMask;
      }
    }
    else if constexpr (target == Target::gpu)
    {
      if constexpr (distrib == Distribution::spst)
      {
        return spst::gpu::supportedBackendMask;
      }
      else if constexpr (distrib == Distribution::spmt)
      {
        return spmt::gpu::supportedBackendMask;
      }
      else if constexpr (distrib == Distribution::mpst)
      {
        return mpst::gpu::supportedBackendMask;
      }
    }

    cxx::unreachable();
  }

  /**
   * @brief Make plan implementation of the specified backend.
   * @tparam BackendParamsT Backend parameters type.
   * @param backend Backend.
   * @param desc Descriptor.
   * @param backendParams Backend parameters.
   * @param feedbackMessage Feedback message.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<Plan>
  makePlan(Backend                                    backend,
           [[maybe_unused]] const Desc&               desc,
           [[maybe_unused]] const BackendParamsT& backendParams,
           std::string*                               feedbackMessage)
  {
    static_assert(isBackendParameters<BackendParamsT>, "Invalid backend parameters type");

    auto assignFeedbackMessage = [&](auto&& message)
    {
      if (feedbackMessage != nullptr)
      {
        *feedbackMessage = std::forward<decltype(message)>(message);
      }
    };

    std::unique_ptr<Plan> plan{};
    
    if ((backend & getSupportedBackendMask<BackendParamsT::target, BackendParamsT::distribution>()) == BackendMask::empty)
    {
      assignFeedbackMessage("Backend not supported for target and distribution");
    }
    else
    {
      try
      {
        switch (backend)
        {
#       if AFFT_BACKEND_IS_ENABLED(CLFFT)
        case Backend::clfft:
          plan = clfft::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(CUFFT)
        case Backend::cufft:
          plan = cufft::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(FFTW3)
        case Backend::fftw3:
          plan = fftw3::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(MKL)
        case Backend::mkl:
          plan = mkl::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(POCKETFFT)
        case Backend::pocketfft:
          plan = pocketfft::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(ROCFFT)
        case Backend::rocfft:
          plan = rocfft::makePlan(desc, backendParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(VKFFT)
        case Backend::vkfft:
          plan = vkfft::makePlan(desc, backendParams);
          break;
#       endif
        default:
          assignFeedbackMessage("Backend is disabled");
          break;
        }
      }
      catch (const std::exception& e)
      {
        assignFeedbackMessage(e.what());
      }
      catch (...)
      {
        assignFeedbackMessage("Unknown error");
      }
    }

    return plan;
  }

  /**
   * @brief Make the first plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Descriptor.
   * @param backendParams Backend parameters.
   * @param feedbacks Feedbacks.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] inline std::unique_ptr<Plan>
  makeFirstPlan(const Desc& desc, const BackendParamsT& backendParams, std::vector<Feedback>* feedbacks)
  {
    std::unique_ptr<Plan> plan{};

    forEachBackend(backendParams.mask, backendParams.order, [&](Backend backend)
    {
      if (!plan)
      {
        std::string* feedbackMessage{};

        if (feedbacks != nullptr)
        {
          auto& feedback = feedbacks->emplace_back();
          feedback.backend = backend;

          feedbackMessage = &feedback.message;
        }
        
        plan = makePlan(backend, desc, backendParams, feedbackMessage);
      }
    });

    return plan;
  }

  /**
   * @brief Make the best plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Descriptor.
   * @param backendParams Backend parameters.
   * @param feedbacks Feedbacks.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] inline std::unique_ptr<Plan>
  makeBestPlan([[maybe_unused]] const Desc& desc,
               [[maybe_unused]] const BackendParamsT& backendParams,
               [[maybe_unused]] std::vector<Feedback>* feedbacks)
  {
    return {};
  }

  /**
   * @brief Make plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Descriptor.
   * @param backendParams Backend parameters.
   * @param feedbacks Feedbacks.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] inline std::unique_ptr<Plan>
  makePlan(const Desc& desc, const BackendParamsT& backendParams, std::vector<Feedback>* feedbacks = nullptr)
  {
    validate(backendParams.strategy);

    std::unique_ptr<Plan> plan{};

    switch (backendParams.strategy)
    {
    case SelectStrategy::first:
      plan = makeFirstPlan(desc, backendParams, feedbacks);
      break;
    case SelectStrategy::best:
      plan = makeBestPlan(desc, backendParams, feedbacks);
      break;
    default:
      cxx::unreachable();
    }

    if (!plan)
    {
      throw std::runtime_error{"Failed to create plan implementation"};
    }

    return plan;
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_HPP */
