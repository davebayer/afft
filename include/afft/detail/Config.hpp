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

#ifndef AFFT_DETAIL_CONFIG_HPP
#define AFFT_DETAIL_CONFIG_HPP

#include <functional>
#include <stdexcept>

#include "common.hpp"
#include "DimensionsConfig.hpp"
#include "TargetConfig.hpp"
#include "TransformConfig.hpp"
#include "utils.hpp"

namespace afft::detail
{
  /**
   * @struct Config
   * @brief Configuration for the transform.
   */
  struct Config
  {
    /**
     * @brief Make a configuration object.
     * @tparam transformType The type of the transform.
     * @tparam target The target platform.
     * @param transformParams The parameters for the transform.
     * @param targetParams The parameters for the target platform.
     * @return The configuration object.
     */
    template<TransformType transformType, Target target>
    [[nodiscard]] static Config make(const typename TransformParametersSelect<transformType>::Type& transformParams,
                                     const typename TargetParametersSelect<target>::Type&           targetParams)
    {
      auto checkCommonParameters = [](const CommonParameters& commonParams)
      {
        switch (commonParams.placement)
        {
        case Placement::inplace: case Placement::outOfPlace: break;
        default: throw makeException<std::invalid_argument>("Invalid placement");
        }

        switch (commonParams.initEffort)
        {
        case InitEffort::low: case InitEffort::med: case InitEffort::high: case InitEffort::max: break;
        default: throw makeException<std::invalid_argument>("Invalid initEffort");
        }

        switch (commonParams.normalize)
        {
        case Normalize::none: case Normalize::orthogonal: case Normalize::unitary: break;
        default: throw makeException<std::invalid_argument>("Invalid normalize");
        }

        switch (commonParams.workspacePolicy)
        {
        case WorkspacePolicy::minimal: case WorkspacePolicy::performance: break;
        default: throw makeException<std::invalid_argument>("Invalid workspacePolicy");
        }
      };

      checkCommonParameters(transformParams.commonParams);

      Config config{.commonParams    = transformParams.commonParams,
                    .dimsConfig      = DimensionsConfig(transformParams.dimensions),
                    .transformConfig = TransformConfig::make(transformParams),
                    .targetConfig    = TargetConfig::make(targetParams)};

      // correct strides in dimensions config
      transformConfig.correctDimensionsConfig(config.dimsConfig, config.getCommonParameters());

      return config;
    }

    CommonParameters commonParams{};  ///< Common parameters.
    DimensionsConfig dimsConfig{};    ///< Dimensions configuration.
    TransformConfig  transformConfig; ///< Transform configuration.
    TargetConfig     targetConfig;    ///< Target configuration.
  };
} // namespace afft::detail

/**
 * @brief Hash specialization for Config.
 */
template<>
struct std::hash<afft::detail::Config>
{
  [[nodiscard]] constexpr std::size_t operator()(const afft::detail::Config& config) const noexcept
  {
    std::size_t seed = 0;

    // TODO

    return seed;
  }
};

#endif /* AFFT_DETAIL_CONFIG_HPP */
