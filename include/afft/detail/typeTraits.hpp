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
  struct TargetParametersSelect<Target::cpu, distrib::Type::single>
  {
    using Type = afft::cpu::Parameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, distrib::Type::single>
  {
    using Type = afft::gpu::Parameters;
  };

  /// @brief Specialization for distributed multi gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu, distrib::Type::multi>
  {
    using Type = afft::multi::gpu::Parameters;
  };

  /// @brief Specialization for distributed mpi cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu, distrib::Type::mpi>
  {
    using Type = afft::mpi::cpu::Parameters;
  };

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename>
  struct IsTargetParameters : std::false_type {};

  /// @brief Specialization for cpu target parameters.
  template<>
  struct IsTargetParameters<afft::cpu::Parameters> : std::true_type {};

  /// @brief Specialization for gpu target parameters.
  template<>
  struct IsTargetParameters<afft::gpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed multi gpu target parameters.
  template<>
  struct IsTargetParameters<afft::multi::gpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed mpi cpu target parameters.
  template<>
  struct IsTargetParameters<afft::mpi::cpu::Parameters> : std::true_type {};

  /// @brief Specialization for distributed cpu target parameters.
  template<>
  struct IsTargetParameters<afft::mpi::gpu::Parameters> : std::true_type {};

  /**
   * @brief ExecutionParameters type for given target.
   * @tparam target The target type.
   * @tparam distribType The distribution type.
   */
  template<Target target, distrib::Type distribType>
  struct TargetExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, distrib::Type::single>
  {
    using Type = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::single>
  {
    using Type = afft::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed multi gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::multi>
  {
    using Type = afft::multi::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpi cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu, distrib::Type::mpi>
  {
    using Type = afft::mpi::cpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpi gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu, distrib::Type::mpi>
  {
    using Type = afft::mpi::gpu::ExecutionParameters;
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

  /// @brief Specialization for distributed multi gpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::multi::gpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for distributed mpi cpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::mpi::cpu::ExecutionParameters> : std::true_type {};

  /// @brief Specialization for distributed mpi gpu ExecutionParameters.
  template<>
  struct IsExecutionParameters<afft::mpi::gpu::ExecutionParameters> : std::true_type {};
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
