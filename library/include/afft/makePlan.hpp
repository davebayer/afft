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
# include <afft/detail/include.hpp>
#endif

#include <afft/Description.hpp>
#include <afft/init.hpp>
#include <afft/memory.hpp>
#include <afft/Plan.hpp>
#include <afft/select.hpp>
#include <afft/detail/makePlan.hpp>

AFFT_EXPORT namespace afft
{
  /**
   * @brief Make a plan.
   * @tparam BackendParamsT Backend parameters.
   * @tparam SelectParamsT Select parameters.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @param selectParams Select parameters.
   * @return The plan.
   */
  template<typename BackendParamsT, typename SelectParamsT = DefaultSelectParameters>
  [[nodiscard]] auto makePlan(const Description&    desc,
                              const BackendParamsT& backendParams,
                              const SelectParamsT&  selectParams = {})
    -> AFFT_RET_REQUIRES(std::unique_ptr<Plan>, isBackendParameters<BackendParamsT> && isSelectParameters<SelectParamsT>)
  {
    if (backendParams.mpBackend != desc.getMpBackend() || backendParams.target != desc.getTarget())
    {
      throw Exception{Error::invalidArgument, "invalid backend parameters"};
    }

    return detail::makePlan(desc, backendParams, selectParams);
  }

  /**
   * @brief Make a plan.
   * @tparam SelectParamsT Select parameters.
   * @param desc Plan description.
   * @param selectParams Select parameters.
   * @return The plan.
   */
  template<typename SelectParamsT = DefaultSelectParameters>
  [[nodiscard]] auto makePlan(const Description&   desc,
                              const SelectParamsT& selectParams = {})
    -> AFFT_RET_REQUIRES(std::unique_ptr<Plan>, isSelectParameters<SelectParamsT>)
  {
    return detail::makePlanWithDefaultBackendParameters(desc, selectParams);
  }

  /**
   * @brief Make a plan.
   * @tparam SelectParamsT Select parameters.
   * @param desc Plan description.
   * @param backendParamsVariant Backend parameters variant.
   * @param selectParams Select parameters.
   * @return The plan.
   */
  template<typename SelectParamsT = DefaultSelectParameters>
  [[nodiscard]] auto makePlan(const Description&              desc,
                              const BackendParametersVariant& backendParamsVariant,
                              const SelectParamsT&            selectParams = {})
    -> AFFT_RET_REQUIRES(std::unique_ptr<Plan>, isSelectParameters<SelectParamsT>)
  {
    return std::visit([&](const auto& backendParams)
    {
      if constexpr (std::is_same_v<std::decay_t<decltype(backendParams)>, std::monostate>)
      {
        return makePlan(desc, selectParams);
      }
      else
      {
        return makePlan(desc, backendParams, selectParams);
      }
    }, backendParamsVariant);
  }
} // namespace afft

#endif /* AFFT_MAKE_PLAN_HPP */
