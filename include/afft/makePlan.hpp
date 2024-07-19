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

#ifndef AFFT_MAKE_PLAN_HPP
#define AFFT_MAKE_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "init.hpp"
#include "memory.hpp"
#include "detail/makePlan.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param targetParams Target parameters.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                          transformParams,
                              const TargetParamsT&                                             targetParams,
                              const BackendParameters<MpBackend::none, TargetParamsT::target>& backendParams = {});

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param targetParams Target parameters.
   * @param memoryLayout Memory layout.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                          transformParams,
                              const TargetParamsT&                                             targetParams,
                              const CentralizedMemoryLayout&                                   memoryLayout,
                              const BackendParameters<MpBackend::none, TargetParamsT::target>& backendParams = {});

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters
   * @tparam MultiProcessParamsT Multi-process parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param multiProcessParams Multi-process parameters.
   * @param targetParams Target parameters.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename MultiProcessParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                                         transformParams,
                              const MultiProcessParamsT&                                                      multiProcessParams,
                              const TargetParamsT&                                                            targetParams,
                              const BackendParameters<MultiProcessParamsT::mpBackend, TargetParamsT::target>& backendParams = {});

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters
   * @tparam MultiProcessParamsT Multi-process parameters.
   * @tparam TargetParamsT Target parameters.
   * @tparam MemoryLayoutT Memory layout.
   * @param transformParams Transform parameters.
   * @param multiProcessParams Multi-process parameters.
   * @param targetParams Target parameters.
   * @param memoryLayout Memory layout.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename MultiProcessParamsT, typename TargetParamsT, typename MemoryLayoutT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                                         transformParams,
                              const MultiProcessParamsT&                                                      multiProcessParams,
                              const TargetParamsT&                                                            targetParams,
                              const MemoryLayoutT&                                                            memoryLayout,
                              const BackendParameters<MultiProcessParamsT::mpBackend, TargetParamsT::target>& backendParams = {});

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param targetParams Target parameters.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                          transformParams,
                              const TargetParamsT&                                             targetParams,
                              const BackendParameters<MpBackend::none, TargetParamsT::target>& backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters");
    static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters");

    return makePlan(transformParams, SingleProcessParameters{}, targetParams, backendParams);
  }

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param targetParams Target parameters.
   * @param memoryLayout Memory layout.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                          transformParams,
                              const TargetParamsT&                                             targetParams,
                              const CentralizedMemoryLayout&                                   memoryLayout,
                              const BackendParameters<MpBackend::none, TargetParamsT::target>& backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters");
    static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters");

    return makePlan(transformParams, SingleProcessParameters{}, targetParams, memoryLayout, backendParams);
  }

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters
   * @tparam MultiProcessParamsT Multi-process parameters.
   * @tparam TargetParamsT Target parameters.
   * @param transformParams Transform parameters.
   * @param multiProcessParams Multi-process parameters.
   * @param targetParams Target parameters.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename MultiProcessParamsT, typename TargetParamsT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                                         transformParams,
                              const MultiProcessParamsT&                                                      multiProcessParams,
                              const TargetParamsT&                                                            targetParams,
                              const BackendParameters<MultiProcessParamsT::mpBackend, TargetParamsT::target>& backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters");
    static_assert(isMpBackendParameters<MultiProcessParamsT>, "invalid multi-process parameters");
    static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters");

    using MemoryLayoutT = std::conditional_t<MultiProcessParamsT::mpBackend == MpBackend::none,
                                             CentralizedMemoryLayout,
                                             DistributedMemoryLayout>;

    return makePlan(transformParams, multiProcessParams, targetParams, MemoryLayoutT{}, backendParams);
  }

  /**
   * @brief Make a plan.
   * @tparam TransformParamsT Transform parameters
   * @tparam MultiProcessParamsT Multi-process parameters.
   * @tparam TargetParamsT Target parameters.
   * @tparam MemoryLayoutT Memory layout.
   * @param transformParams Transform parameters.
   * @param multiProcessParams Multi-process parameters.
   * @param targetParams Target parameters.
   * @param memoryLayout Memory layout.
   * @param backendParams Backend parameters.
   * @return The plan.
   */
  template<typename TransformParamsT, typename MultiProcessParamsT, typename TargetParamsT, typename MemoryLayoutT>
  [[nodiscard]] auto makePlan(const TransformParamsT&                                                         transformParams,
                              const MultiProcessParamsT&                                                      multiProcessParams,
                              const TargetParamsT&                                                            targetParams,
                              const MemoryLayoutT&                                                            memoryLayout,
                              const BackendParameters<MultiProcessParamsT::mpBackend, TargetParamsT::target>& backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters");
    static_assert(isMpBackendParameters<MultiProcessParamsT>, "invalid multi-process parameters");
    static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters");
    static_assert(isMemoryLayout<MemoryLayoutT>, "invalid memory layout");

    static_assert(MultiProcessParamsT::mpBackend != MpBackend::none || std::is_same_v<MemoryLayoutT, CentralizedMemoryLayout>,
                  "distributed memory layout is only supported for multi-process backends");

    init();

    const auto& desc = detail::Desc{transformParams, multiProcessParams, targetParams, memoryLayout, backendParams};

    return detail::makePlan(desc, backendParams);
  }
} // namespace afft

#endif /* AFFT_MAKE_PLAN_HPP */
