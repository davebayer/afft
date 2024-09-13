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
#include "../target.hpp"

#ifdef AFFT_ENABLE_CLFFT
# include "clfft/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_CUFFT
# include "cufft/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_FFTW3
# include "fftw3/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_HEFFTE
# include "heffte/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_HIPFFT
# include "hipfft/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_MKL
# include "mkl/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_POCKETFFT
# include "pocketfft/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_ROCFFT
# include "rocfft/makePlan.hpp"
#endif
#ifdef AFFT_ENABLE_VKFFT
# include "vkfft/makePlan.hpp"
#endif

namespace afft::detail
{
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
    for (std::size_t i{}; i < backendCount; ++i)
    {
      const Backend backend = static_cast<Backend>(i);

      if ((backendMask & makeBackendMask(backend)) != BackendMask::empty)
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
      if ((backendMask & makeBackendMask(backend)) != BackendMask::empty)
      {
        fn(backend);
      }

      backendMask = backendMask & (~makeBackendMask(backend));
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
   * @param targetCount Target count.
   * @return Supported backend mask.
   */
  template<MpBackend mpBackend, Target target>
  [[nodiscard]] constexpr BackendMask getSupportedBackendMask(std::size_t targetCount)
  {
    static_assert(isValid(mpBackend), "invalid multi process backend");
    static_assert(isValid(target), "invalid target");

    if constexpr (mpBackend == MpBackend::none)
    {
      if constexpr (target == Target::cpu)
      {
        return BackendMask::fftw3 | BackendMask::mkl | BackendMask::pocketfft;
      }
      else if constexpr (target == Target::cuda)
      {
        return BackendMask::cufft | ((targetCount == 1) ? (BackendMask::empty | BackendMask::vkfft) : BackendMask::empty);
      }
      else if constexpr (target == Target::hip)
      {
        return BackendMask::hipfft | BackendMask::rocfft | ((targetCount == 1) ? (BackendMask::empty | BackendMask::vkfft) : BackendMask::empty);
      }
      else if constexpr (target == Target::opencl)
      {
        return (targetCount == 1) ? (BackendMask::clfft | BackendMask::vkfft) : BackendMask::empty;
      }
    }
    else if constexpr (mpBackend == MpBackend::mpi)
    {
      if constexpr (target == Target::cpu)
      {
        return BackendMask::fftw3 | BackendMask::heffte | BackendMask::mkl;
      }
      else if constexpr (target == Target::cuda)
      {
        return (targetCount == 1) ? (BackendMask::cufft | BackendMask::heffte) : BackendMask::empty;
      }
      else if constexpr (target == Target::hip)
      {
        return BackendMask::rocfft | ((targetCount == 1) ? (BackendMask::hipfft | BackendMask::heffte) : BackendMask::empty);
      }
      else if constexpr (target == Target::opencl)
      {
        return BackendMask::empty;
      }
    }

    cxx::unreachable();
  }

  /**
   * @brief Make plan implementation of the specified backend.
   * @tparam BackendParamsT Backend parameters type.
   * @param backend Backend.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @param feedbackMessage Feedback message.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(Backend                                backend,
           [[maybe_unused]] const Description&    desc,
           [[maybe_unused]] const BackendParamsT& backendParams,
           std::string*                           feedbackMessage)
  {
    static_assert(isBackendParameters<BackendParamsT>, "Invalid backend parameters type");

    auto assignFeedbackMessage = [&](auto&& message)
    {
      if (feedbackMessage != nullptr)
      {
        *feedbackMessage = std::forward<decltype(message)>(message);
      }
    };

    std::unique_ptr<afft::Plan> plan{};

    const auto supportedBackendMask = getSupportedBackendMask<BackendParamsT::mpBackend, BackendParamsT::target>(desc.getTargetCount());
    
    if ((makeBackendMask(backend) & supportedBackendMask) == BackendMask::empty)
    {
      assignFeedbackMessage("Backend not supported for target and distribution");
    }
    else
    {
      try
      {
        switch (backend)
        {
#       ifdef AFFT_ENABLE_CLFFT
        case Backend::clfft:
          plan = clfft::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_CUFFT
        case Backend::cufft:
          plan = cufft::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_FFTW3
        case Backend::fftw3:
          plan = fftw3::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_HEFFTE
        case Backend::heffte:
          plan = heffte::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_HIPFFT
        case Backend::hipfft:
          plan = hipfft::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_MKL
        case Backend::mkl:
          plan = mkl::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_POCKETFFT
        case Backend::pocketfft:
          plan = pocketfft::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_ROCFFT
        case Backend::rocfft:
          plan = rocfft::makePlan(desc, backendParams);
          break;
#       endif
#       ifdef AFFT_ENABLE_VKFFT
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
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @param feedbacks Feedbacks.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description&           desc,
           const BackendParamsT&        backendParams,
           const FirstSelectParameters& selectParams)
  {
    std::unique_ptr<afft::Plan> plan{};

    forEachBackend(selectParams.mask, selectParams.order, [&](Backend backend)
    {
      if (!plan)
      {
        std::string* feedbackMessage{};

        // if (feedbacks != nullptr)
        // {
        //   auto& feedback = feedbacks->emplace_back();
        //   feedback.backend = backend;

        //   feedbackMessage = &feedback.message;
        // }
        
        plan = makePlan(backend, desc, backendParams, feedbackMessage);
      }
    });

    if (!plan)
    {
      throw Exception{Error::internal, "No plan implementation found"};
    }

    return plan;
  }

  /**
   * @brief Make the best plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @param feedbacks Feedbacks.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makeBestPlan([[maybe_unused]] const Description&          desc,
               [[maybe_unused]] const BackendParamsT&       backendParams,
               [[maybe_unused]] const BestSelectParameters& selectParams)
  {
    throw Exception{Error::internal, "Best plan selection is not implemented"};
  }

  /**
   * @brief Make plan with default backend parameters helper.
   * @tparam mpBackend Multi-process backend.
   * @tparam SelectParamsT Select parameters type.
   * @param desc Plan description.
   * @param selectParams Select parameters.
   * @return Plan.
   */
  template<MpBackend mpBackend, typename SelectParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlanWithDefaultBackendParametersHelper(const Description& desc, const SelectParamsT& selectParams)
  {
    switch (desc.getTarget())
    {
    case Target::cpu:
#   ifdef AFFT_ENABLE_CPU
      return detail::makePlan(desc, BackendParameters<mpBackend, Target::cpu>{}, selectParams);
#   else
      throw Exception{Error::invalidArgument, "CPU backend is not enabled"};
#   endif
    case Target::cuda:
#   ifdef AFFT_ENABLE_CUDA
      return detail::makePlan(desc, BackendParameters<mpBackend, Target::cuda>{}, selectParams);
#   else
      throw Exception{Error::invalidArgument, "CUDA backend is not enabled"};
#   endif
    case Target::hip:
#   ifdef AFFT_ENABLE_HIP
      return detail::makePlan(desc, BackendParameters<mpBackend, Target::hip>{}, selectParams);
#   else
      throw Exception{Error::invalidArgument, "HIP backend is not enabled"};
#   endif
    case Target::opencl:
#   ifdef AFFT_ENABLE_OPENCL
      return detail::makePlan(desc, BackendParameters<mpBackend, Target::opencl>{}, selectParams);
#   else
      throw Exception{Error::invalidArgument, "OpenCL backend is not enabled"};
#   endif
    case Target::openmp:
#   ifdef AFFT_ENABLE_OPENMP
      return detail::makePlan(desc, BackendParameters<mpBackend, Target::openmp>{}, selectParams);
#   else
      throw Exception{Error::invalidArgument, "OpenMP backend is not enabled"};
#   endif
    default:
      cxx::unreachable();
    }
  }

  /**
   * @brief Make plan with default backend parameters.
   * @tparam SelectParamsT Select parameters type.
   * @param desc Plan description.
   * @param selectParams Select parameters.
   * @return Plan.
   */
  template<typename SelectParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlanWithDefaultBackendParameters(const Description& desc, const SelectParamsT& selectParams)
  {
    switch (desc.getMpBackend())
    {
    case MpBackend::none:
      return makePlanWithDefaultBackendParametersHelper<MpBackend::none>(desc, selectParams);
    case MpBackend::mpi:
#   ifdef AFFT_ENABLE_MPI
      return makePlanWithDefaultBackendParametersHelper<MpBackend::mpi>(desc, selectParams);
#   else
      throw Exception{Error::invalidArgument, "MPI backend is not enabled"};
#   endif
    default:
      cxx::unreachable();
    }
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_HPP */
