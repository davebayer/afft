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

#include "detail/makePlan.hpp"

namespace afft
{
  /**
   * @brief Create a plan for the given transform, architecture and backend parameters
   * @tparam TransformParamsT Transform parameters type
   * @tparam ArchParamsT Architecture parameters type
   * @tparam BackendParamsT Backend parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @param backendParams Backend parameters
   * @return Plan
   */
  template<typename TransformParamsT, typename ArchParamsT, typename BackendParamsT>
  std::unique_ptr<Plan> makePlan(const TransformParamsT& transformParams,
                                 ArchParamsT&            archParams,
                                 const BackendParamsT&   backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParamsT>, "Invalid architecture parameters type");
    static_assert(detail::isKnownBackendParams<BackendParamsT>, "Invalid backend parameters type");

    // static_assert(detail::isCompatible<ArchParamsT, BackendParamsT>,
    //              "Architecture and backend parameters must share the same target and distribution");

    static constexpr auto transformParamsShapeRank = detail::TransformParametersTemplateRanks<TransformParamsT>::shape;
    static constexpr auto archParamsShapeRank      = detail::ArchParametersTemplateRanks<ArchParamsT>::shape;

    static_assert((transformParamsShapeRank == dynamicRank) ||
                  (archParamsShapeRank == dynamicRank) ||
                  (transformParamsShapeRank == archParamsShapeRank),
                  "Transform and target parameters must have the same shape rank");

    return detail::makePlan(detail::Desc{transformParams, archParams}, backendParams);
  }

  /**
   * @brief Create a plan with feedback for the given transform, architecture and backend parameters
   * @tparam TransformParamsT Transform parameters type
   * @tparam ArchParamsT Architecture parameters type
   * @tparam BackendParamsT Backend parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @param backendParams Backend parameters
   * @return Plan and feedback
   */
  template<typename TransformParamsT, typename ArchParamsT, typename BackendParamsT>
  std::pair<std::unique_ptr<Plan>, std::vector<Feedback>>
  makePlanWithFeedback(const TransformParamsT& transformParams,
                       ArchParamsT&            archParams,
                       const BackendParamsT&   backendParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParamsT>, "Invalid architecture parameters type");
    static_assert(detail::isKnownBackendParams<BackendParamsT>, "Invalid backend parameters type");

    // static_assert(detail::isCompatible<ArchParamsT, BackendParamsT>,
    //              "Architecture and backend parameters must share the same target and distribution");

    static constexpr auto transformParamsShapeRank = detail::TransformParametersTemplateRanks<TransformParamsT>::shape;
    static constexpr auto archParamsShapeRank      = detail::ArchParametersTemplateRanks<ArchParamsT>::shape;

    static_assert((transformParamsShapeRank == dynamicRank) ||
                  (archParamsShapeRank == dynamicRank) ||
                  (transformParamsShapeRank == archParamsShapeRank),
                  "Transform and target parameters must have the same shape rank");

    std::pair<std::unique_ptr<Plan>, std::vector<Feedback>> result{};

    result.first = detail::makePlan(detail::Desc{transformParams, archParams}, backendParams, &result.second);

    return result;
  }
} // namespace afft

#endif /* AFFT_MAKE_PLAN_HPP */
