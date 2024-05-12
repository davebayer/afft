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

#include <type_traits>

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
    using Type = dft::Parameters;
  };

  /// @brief Specialization for dht transform.
  template<>
  struct TransformParametersSelect<Transform::dht>
  {
    using Type = dht::Parameters;
  };

  /// @brief Specialization for dtt transform.
  template<>
  struct TransformParametersSelect<Transform::dtt>
  {
    using Type = dtt::Parameters;
  };

  /**
   * @brief Check if the type is TransformParameters.
   * @tparam T The type.
   */
  template<typename>
  struct IsTransformParameters : std::false_type {};

  /// @brief Specialization for dft transform parameters.
  template<>
  struct IsTransformParameters<afft::dft::Parameters> : std::true_type {};

  /// @brief Specialization for dtt transform parameters.
  template<>
  struct IsTransformParameters<afft::dtt::Parameters> : std::true_type {};

  /**
   * @brief Transform backend type for given target.
   * @tparam target The target type.
   * @tparam distribType The distribution type.
   */
  template<Target target, distrib::Type distribType>
  struct TargetParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu, distrib::Type::spst>
  {
    using Type = afft::cpu::Parameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, distrib::Type::spst>
  {
    using Type = afft::gpu::Parameters;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, distrib::Type::spmt>
  {
    using Type = afft::spmt::gpu::Parameters;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu, distrib::Type::mpst>
  {
    using Type = afft::mpst::cpu::Parameters;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, distrib::Type::mpst>
  {
    using Type = afft::mpst::gpu::Parameters;
  };

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename>
  struct IsTargetParameters : std::false_type {};

  /// @brief Specialization for cpu target parameters.
  template<>
  struct IsTargetParameters<afft::spst::cpu::Parameters> : std::true_type {};

  /// @brief Specialization for gpu target parameters.
  template<>
  struct IsTargetParameters<afft::spst::gpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed spmt gpu target parameters.
  template<>
  struct IsTargetParameters<afft::spmt::gpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed mpst cpu target parameters.
  template<>
  struct IsTargetParameters<afft::mpst::cpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed cpu target parameters.
  template<>
  struct IsTargetParameters<afft::mpst::gpu::Parameters> : std::true_type {};

  /**
   * @brief ExecutionParameters type for given target.
   * @tparam target The target type.
   * @tparam distribType The distribution type.
   */
  template<Target target, distrib::Type distribType>
  struct TargetExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, distrib::Type::spst>
  {
    using Type = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::spst>
  {
    using Type = afft::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::spmt>
  {
    using Type = afft::spmt::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, distrib::Type::mpst>
  {
    using Type = afft::mpst::cpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::mpst>
  {
    using Type = afft::mpst::gpu::ExecutionParameters;
  };

  /**
   * @brief Check if the type is ExecutionParameters.
   * @tparam T The type.
   */
  template<typename>
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
