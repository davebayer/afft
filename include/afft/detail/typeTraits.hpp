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
#include "../type.hpp"

namespace afft::detail
{
inline namespace cxx20
{
  template<typename T>
  struct remove_cvref
  {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  template<typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;
} // namespace cxx20

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
   * @brief Target Parameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  struct TargetParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu>
  {
    using Type = afft::cpu::Parameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetParametersSelect<Target::gpu>
  {
    using Type = afft::gpu::Parameters;
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

  /**
   * @brief Transform backend type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  struct TargetBackendSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetBackendSelect<Target::cpu>
  {
    using Type = afft::cpu::Backend;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetBackendSelect<Target::gpu>
  {
    using Type = afft::gpu::Backend;
  };

  /**
   * @brief Check if the type is target backend.
   * @tparam T The type.
   */
  template<typename>
  struct IsTargetBackend : std::false_type {};

  /// @brief Specialization for cpu backend.
  template<>
  struct IsTargetBackend<afft::cpu::Backend> : std::true_type {};

  /// @brief Specialization for gpu backend.
  template<>
  struct IsTargetBackend<afft::gpu::Backend> : std::true_type {};

  /**
   * @brief BackendSelectParameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  struct TargetBackendSelectParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetBackendSelectParametersSelect<Target::cpu>
  {
    using Type = afft::cpu::BackendSelectParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetBackendSelectParametersSelect<Target::gpu>
  {
    using Type = afft::gpu::BackendSelectParameters;
  };

  /**
   * @brief Check if the type is BackendSelectParameters.
   * @tparam T The type.
   */
  template<typename>
  struct IsBackendSelectParameters : std::false_type {};

  /// @brief Specialization for cpu BackendSelectParameters.
  template<>
  struct IsBackendSelectParameters<afft::cpu::BackendSelectParameters> : std::true_type {};

  /// @brief Specialization for gpu BackendSelectParameters.
  template<>
  struct IsBackendSelectParameters<afft::gpu::BackendSelectParameters> : std::true_type {};

  /**
   * @brief Get the target of the backend select parameters.
   * @tparam T The type.
   */
  template<typename T>
  struct BackendSelectParametersTarget;

  /// @brief Specialization for cpu target.
  template<>
  struct BackendSelectParametersTarget<afft::cpu::BackendSelectParameters>
  {
    static constexpr Target value{Target::cpu};
  };

  /// @brief Specialization for gpu target.
  template<>
  struct BackendSelectParametersTarget<afft::gpu::BackendSelectParameters>
  {
    static constexpr Target value{Target::gpu};
  };

  /**
   * @brief ExecutionParameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  struct TargetExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::cpu>
  {
    using Type = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetExecutionParametersSelect<Target::gpu>
  {
    using Type = afft::gpu::ExecutionParameters;
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

  /**
   * @brief Get the target of the target parameters.
   * @tparam T The type.
   */
  template<typename T>
  struct TargetParametersTarget;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetParametersTarget<afft::cpu::Parameters>
  {
    static constexpr Target value{Target::cpu};
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetParametersTarget<afft::gpu::Parameters>
  {
    static constexpr Target value{Target::gpu};
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
