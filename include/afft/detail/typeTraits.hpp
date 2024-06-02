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

#ifndef AFFT_DETAIL_TYPE_TRAITS_HPP
#define AFFT_DETAIL_TYPE_TRAITS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "../cpu.hpp"
#include "../distrib.hpp"
#include "../gpu.hpp"
#include "../type.hpp"

namespace afft::detail
{
  /**
   * @brief TypeProperties helper. Removes const and volatile from Complex template parameter type.
   * @tparam T The type.
   */
  template<typename T>
  struct TypePropertiesHelper
    : TypeProperties<std::remove_cv_t<T>> {};
  
  /// @brief Specialization for Complex.
  template<typename T>
  struct TypePropertiesHelper<Complex<T>>
    : TypeProperties<Complex<std::remove_cv_t<T>>> {};

  /**
   * @brief Target Parameters type for given target.
   * @tparam target The target type.
   */
  template<Transform transform>
  struct TransformParametersSelect;

  /// @brief Specialization for dft transform.
  template<>
  struct TransformParametersSelect<Transform::dft>
  {
    using Type = dft::Parameters<>;
  };

  /// @brief Specialization for dht transform.
  template<>
  struct TransformParametersSelect<Transform::dht>
  {
    using Type = dht::Parameters<>;
  };

  /// @brief Specialization for dtt transform.
  template<>
  struct TransformParametersSelect<Transform::dtt>
  {
    using Type = dtt::Parameters<>;
  };

  /**
   * @brief Check if the type is TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsTransformParameters : std::false_type {};

  /// @brief Specialization for dft transform parameters.
  template<std::size_t sRank, std::size_t tRank>
  struct IsTransformParameters<afft::dft::Parameters<sRank, tRank>> : std::true_type {};

  /// @brief Specialization for dht transform parameters.
  template<std::size_t sRank, std::size_t tRank>
  struct IsTransformParameters<afft::dht::Parameters<sRank, tRank>> : std::true_type {};

  /// @brief Specialization for dtt transform parameters.
  template<std::size_t sRank, std::size_t tRank, std::size_t ttRank>
  struct IsTransformParameters<afft::dtt::Parameters<sRank, tRank, ttRank>> : std::true_type {};

  /**
   * @brief TransformParameters template ranks.
   * @tparam T The transform parameters type.
   */
  template<typename T>
  struct TransformParametersTemplateRanks;

  /// @brief Specialization for dft transform parameters.
  template<std::size_t sRank, std::size_t tRank>
  struct TransformParametersTemplateRanks<afft::dft::Parameters<sRank, tRank>>
  {
    static constexpr std::size_t shape     = sRank;
    static constexpr std::size_t transform = tRank;
  };
  
  /// @brief Specialization for dht transform parameters.
  template<std::size_t sRank, std::size_t tRank>
  struct TransformParametersTemplateRanks<afft::dht::Parameters<sRank, tRank>>
  {
    static constexpr std::size_t shape     = sRank;
    static constexpr std::size_t transform = tRank;
  };

  /// @brief Specialization for dtt transform parameters.
  template<std::size_t sRank, std::size_t tRank, std::size_t ttRank>
  struct TransformParametersTemplateRanks<afft::dtt::Parameters<sRank, tRank, ttRank>>
  {
    static constexpr std::size_t shape         = sRank;
    static constexpr std::size_t transform     = tRank;
    static constexpr std::size_t transformType = ttRank;
  };

  /**
   * @brief Transform backend type for given target.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib>
  struct TargetParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu, Distribution::spst>
  {
    using Type = afft::cpu::Parameters<>;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, Distribution::spst>
  {
    using Type = afft::gpu::Parameters<>;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, Distribution::spmt>
  {
    using Type = afft::spmt::gpu::Parameters<>;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu, Distribution::mpst>
  {
    using Type = afft::mpst::cpu::Parameters<>;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, Distribution::mpst>
  {
    using Type = afft::mpst::gpu::Parameters<>;
  };

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsTargetParameters : std::false_type {};

  /// @brief Specialization for cpu target parameters.
  template<std::size_t sRank>
  struct IsTargetParameters<afft::spst::cpu::Parameters<sRank>> : std::true_type {};

  /// @brief Specialization for gpu target parameters.
  template<std::size_t sRank>
  struct IsTargetParameters<afft::spst::gpu::Parameters<sRank>> : std::true_type {};

  /// @brief Specialization for distributed spmt gpu target parameters.
  template<std::size_t sRank>
  struct IsTargetParameters<afft::spmt::gpu::Parameters<sRank>> : std::true_type {};

  /// @brief Specialization for distributed mpst cpu target parameters.
  template<std::size_t sRank>
  struct IsTargetParameters<afft::mpst::cpu::Parameters<sRank>> : std::true_type {};

  /// @brief Specialization for distributed cpu target parameters.
  template<std::size_t sRank>
  struct IsTargetParameters<afft::mpst::gpu::Parameters<sRank>> : std::true_type {};

  /**
   * @brief TargetParameters template ranks.
   * @tparam T The target parameters type.
   */
  template<typename T>
  struct TargetParametersTemplateRanks;

  /// @brief Specialization for cpu target parameters.
  template<std::size_t sRank>
  struct TargetParametersTemplateRanks<afft::spst::cpu::Parameters<sRank>>
  {
    static constexpr std::size_t shape = sRank;
  };

  /// @brief Specialization for gpu target parameters.
  template<std::size_t sRank>
  struct TargetParametersTemplateRanks<afft::spst::gpu::Parameters<sRank>>
  {
    static constexpr std::size_t shape = sRank;
  };

  /// @brief Specialization for distributed spmt gpu target parameters.
  template<std::size_t sRank>
  struct TargetParametersTemplateRanks<afft::spmt::gpu::Parameters<sRank>>
  {
    static constexpr std::size_t shape = sRank;
  };

  /// @brief Specialization for distributed mpst cpu target parameters.
  template<std::size_t sRank>
  struct TargetParametersTemplateRanks<afft::mpst::cpu::Parameters<sRank>>
  {
    static constexpr std::size_t shape = sRank;
  };

  /// @brief Specialization for distributed mpst gpu target parameters.
  template<std::size_t sRank>
  struct TargetParametersTemplateRanks<afft::mpst::gpu::Parameters<sRank>>
  {
    static constexpr std::size_t shape = sRank;
  };  

  /**
   * @brief ExecutionParameters type for given target.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib>
  struct TargetExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, Distribution::spst>
  {
    using Type = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, Distribution::spst>
  {
    using Type = afft::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, Distribution::spmt>
  {
    using Type = afft::spmt::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, Distribution::mpst>
  {
    using Type = afft::mpst::cpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, Distribution::mpst>
  {
    using Type = afft::mpst::gpu::ExecutionParameters;
  };

  /**
   * @brief Check if the type is SelectParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsSelectParameters : std::false_type {};

  /// @brief Specialization for spst cpu SelectParameters.
  template<>
  struct IsSelectParameters<afft::spst::cpu::SelectParameters> : std::true_type {};

  /// @brief Specialization for spst gpu SelectParameters.
  template<>
  struct IsSelectParameters<afft::spst::gpu::SelectParameters> : std::true_type {};

  /// @brief Specialization for spmt gpu SelectParameters.
  template<>
  struct IsSelectParameters<afft::spmt::gpu::SelectParameters> : std::true_type {};

  /// @brief Specialization for mpst cpu SelectParameters.
  template<>
  struct IsSelectParameters<afft::mpst::cpu::SelectParameters> : std::true_type {};

  /// @brief Specialization for mpst gpu SelectParameters.
  template<>
  struct IsSelectParameters<afft::mpst::gpu::SelectParameters> : std::true_type {};

  /**
   * @brief Check if the type is ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsExecutionParameters : std::false_type {};

  /// @brief Specialization for cpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::cpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for gpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::gpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for distributed spmt gpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::spmt::gpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for distributed mpst cpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::mpst::cpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for distributed mpst gpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::mpst::gpu::ExecutionParameters> : std::true_type {};
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
