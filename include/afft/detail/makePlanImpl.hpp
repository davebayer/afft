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
#if AFFT_BACKEND_IS_ENABLED(CLFFT)
# include "clfft/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(CUFFT)
# include "cufft/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(FFTW3)
# include "fftw3/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(MKL)
# include "mkl/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(POCKETFFT)
# include "pocketfft/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(ROCFFT)
# include "rocfft/makePlanImpl.hpp"
#endif
#if AFFT_BACKEND_IS_ENABLED(VKFFT)
# include "vkfft/makePlanImpl.hpp"
#endif

namespace afft::detail
{
  

  template<typename Fn>
  auto forEachBackend(BackendMask backendMask, Fn&& fn)
    -> AFFT_RET_REQUIRES(void, AFFT_PARAM(std::is_invocable_v<Fn, Backend>))
  {
    for (BackendUnderlyingType i{}; i < cxx::to_underlying(Backend::_count); ++i)
    {
      const Backend backend = static_cast<Backend>(i);

      if (backendMask & backend)
      {
        fn(backend);
      }
    }
  }

  template<typename Fn>
  auto forEachBackend(BackendMask backendMask, View<Backend> backendOrder, Fn&& fn)
    -> AFFT_RET_REQUIRES(void, AFFT_PARAM(std::is_invocable_v<Fn, Backend>))
  {
    for (const Backend backend : backendOrder)
    {
      if (backendMask & backend)
      {
        fn(backend);
      }

      backendMask &= ~backend;
    }

    if (cxx::to_underlying(backendMask))
    {
      forEachBackend(backendMask, std::forward<Fn>(fn));
    }
  }

  template<Target target, Distribution distrib>
  [[nodiscard]] std::unique_ptr<PlanImpl>
  makePlanImpl(Backend                                  backend,
               const Desc&                              desc,
               const SelectParameters<target, distrib>& selectParams,
               std::string*                             feedbackMessage)
  {
    auto assignFeedbackMessage = [&](auto&& message)
    {
      if (feedbackMessage != nullptr)
      {
        *feedbackMessage = std::forward<decltype(message)>(message);
      }
    };

    std::unique_ptr<PlanImpl> planImpl{};
    
    if (!(backend & supportedBackendMask<target, distrib>))
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
          planImpl = clfft::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(CUFFT)
        case Backend::cufft:
          planImpl = cufft::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(FFTW3)
        case Backend::fftw3:
          planImpl = fftw3::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(MKL)
        case Backend::mkl:
          planImpl = mkl::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(POCKETFFT)
        case Backend::pocketfft:
          planImpl = pocketfft::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(ROCFFT)
        case Backend::rocfft:
          planImpl = rocfft::makePlanImpl(desc, selectParams);
          break;
#       endif
#       if AFFT_BACKEND_IS_ENABLED(VKFFT)
        case Backend::vkfft:
          planImpl = vkfft::makePlanImpl(desc, selectParams);
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

    return planImpl;
  }

  template<typename SelectParametersT>
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makeFirstPlanImpl(const Desc& desc, const SelectParametersT& selectParams, std::vector<Feedback>* feedbacks)
  {
    std::unique_ptr<PlanImpl> planImpl{};

    forEachBackend(selectParams.mask, selectParams.order, [&](Backend backend)
    {
      if (!planImpl)
      {
        std::string* feedbackMessage{};

        if (feedbacks != nullptr)
        {
          auto& feedback = feedbacks->emplace_back();
          feedback.backend = backend;

          feedbackMessage = &feedback.message;
        }
        
        planImpl = makePlanImpl(backend, desc, selectParams, feedbackMessage);
      }
    });

    return planImpl;
  }

  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makeBestPlanImpl(const Desc& desc, const BackendParameters& backendParams, std::vector<Feedback>* feedbacks)
  {
    return {};
  }

  template<typename SelectParametersT>
  [[nodiscard]] inline std::unique_ptr<PlanImpl>
  makePlanImpl(const Desc& desc, const SelectParametersT& selectParams, std::vector<Feedback>* feedbacks = nullptr)
  {
    validate(selectParams.strategy);

    std::unique_ptr<PlanImpl> planImpl{};

    switch (selectParams.selectStrategy)
    {
    case SelectStrategy::first:
      planImpl = makeFirstPlanImpl(desc, selectParams, feedbacks);
      break;
    case SelectStrategy::best:
      planImpl = makeBestPlanImpl(desc, selectParams, feedbacks);
      break;
    default:
      cxx::unreachable();
    }

    if (!planImpl)
    {
      throw makeException<std::runtime_error>("Failed to create plan implementation");
    }

    return planImpl;
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_IMPL_HPP */
