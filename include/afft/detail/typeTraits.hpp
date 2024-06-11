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
#include "../architecture.hpp"
#include "../transform.hpp"
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
  
  /// @brief Specialization for std::complex.
  template<typename T>
  struct TypePropertiesHelper<std::complex<T>>
    : TypeProperties<std::complex<std::remove_cv_t<T>>> {};

  /**
   * @brief TransformParameters type for given transform.
   * @tparam transform The transform type.
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
  template<std::size_t shapeExt, std::size_t transformExt>
  struct IsTransformParameters<afft::dft::Parameters<shapeExt, transformExt>> : std::true_type {};

  /// @brief Specialization for dht transform parameters.
  template<std::size_t shapeExt, std::size_t transformExt>
  struct IsTransformParameters<afft::dht::Parameters<shapeExt, transformExt>> : std::true_type {};

  /// @brief Specialization for dtt transform parameters.
  template<std::size_t shapeExt, std::size_t transformExt>
  struct IsTransformParameters<afft::dtt::Parameters<shapeExt, transformExt>> : std::true_type {};

  /**
   * @brief ArchitectureParameters type selection based on given target and distribution.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib>
  struct ArchParametersSelect;

  /// @brief Specialization for spst cpu target.
  template<>
  struct ArchParametersSelect<Target::cpu, Distribution::spst>
  {
    using Type = afft::cpu::Parameters<>;
  };

  /// @brief Specialization for spst gpu target.
  template<>
  struct ArchParametersSelect<Target::gpu, Distribution::spst>
  {
    using Type = afft::gpu::Parameters<>;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct ArchParametersSelect<Target::gpu, Distribution::spmt>
  {
    using Type = afft::spmt::gpu::Parameters<>;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct ArchParametersSelect<Target::cpu, Distribution::mpst>
  {
    using Type = afft::mpst::cpu::Parameters<>;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct ArchParametersSelect<Target::gpu, Distribution::mpst>
  {
    using Type = afft::mpst::gpu::Parameters<>;
  };

  /**
   * @brief Check if the type is ArchitectureParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsArchParameters : std::false_type {};

  /// @brief Specialization for cpu target parameters.
  template<std::size_t shapeExt>
  struct IsArchParameters<afft::spst::cpu::Parameters<shapeExt>> : std::true_type {};

  /// @brief Specialization for gpu target parameters.
  template<std::size_t shapeExt>
  struct IsArchParameters<afft::spst::gpu::Parameters<shapeExt>> : std::true_type {};

  /// @brief Specialization for distributed spmt gpu target parameters.
  template<std::size_t shapeExt>
  struct IsArchParameters<afft::spmt::gpu::Parameters<shapeExt>> : std::true_type {};

  /// @brief Specialization for distributed mpst cpu target parameters.
  template<std::size_t shapeExt>
  struct IsArchParameters<afft::mpst::cpu::Parameters<shapeExt>> : std::true_type {};

  /// @brief Specialization for distributed cpu target parameters.
  template<std::size_t shapeExt>
  struct IsArchParameters<afft::mpst::gpu::Parameters<shapeExt>> : std::true_type {};

  /**
   * @brief BackendParameters type for given architecture.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib>
  struct BackendParametersSelect;

  /// @brief Specialization for spst cpu target.
  template<>
  struct BackendParametersSelect<Target::cpu, Distribution::spst>
  {
    using Type = afft::spst::cpu::BackendParameters;
  };

  /// @brief Specialization for spst gpu target.
  template<>
  struct BackendParametersSelect<Target::gpu, Distribution::spst>
  {
    using Type = afft::spst::gpu::BackendParameters;
  };

  /// @brief Specialization for spmt gpu target.
  template<>
  struct BackendParametersSelect<Target::gpu, Distribution::spmt>
  {
    using Type = afft::spmt::gpu::BackendParameters;
  };

  /// @brief Specialization for mpst cpu target.
  template<>
  struct BackendParametersSelect<Target::cpu, Distribution::mpst>
  {
    using Type = afft::mpst::cpu::BackendParameters;
  };

  /// @brief Specialization for mpst gpu target.
  template<>
  struct BackendParametersSelect<Target::gpu, Distribution::mpst>
  {
    using Type = afft::mpst::gpu::BackendParameters;
  };

  /**
   * @brief Check if the type is BackendParametersSelect.
   * @tparam T The type.
   */
  template<typename T>
  struct IsBackendParameters
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, afft::spst::cpu::BackendParameters> ||
                         std::is_same_v<std::remove_cv_t<T>, afft::spst::gpu::BackendParameters> ||
                         std::is_same_v<std::remove_cv_t<T>, afft::spmt::gpu::BackendParameters> ||
                         std::is_same_v<std::remove_cv_t<T>, afft::mpst::cpu::BackendParameters> ||
                         std::is_same_v<std::remove_cv_t<T>, afft::mpst::gpu::BackendParameters>> {};

  /**
   * @brief ExecutionParameters type for given architecture.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib>
  struct ArchExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct ArchExecutionParametersSelect<Target::cpu, Distribution::spst>
  {
    using Type = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct ArchExecutionParametersSelect<Target::gpu, Distribution::spst>
  {
    using Type = afft::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed spmt gpu target.
  template<>
  struct ArchExecutionParametersSelect<Target::gpu, Distribution::spmt>
  {
    using Type = afft::spmt::gpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst cpu target.
  template<>
  struct ArchExecutionParametersSelect<Target::cpu, Distribution::mpst>
  {
    using Type = afft::mpst::cpu::ExecutionParameters;
  };

  /// @brief Specialization for distributed mpst gpu target.
  template<>
  struct ArchExecutionParametersSelect<Target::gpu, Distribution::mpst>
  {
    using Type = afft::mpst::gpu::ExecutionParameters;
  };

  /**
   * @brief Check if the type is ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsExecutionParameters
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, afft::spst::cpu::ExecutionParameters> || 
                         std::is_same_v<std::remove_cv_t<T>, afft::spst::gpu::ExecutionParameters> || 
                         std::is_same_v<std::remove_cv_t<T>, afft::spmt::gpu::ExecutionParameters> || 
                         std::is_same_v<std::remove_cv_t<T>, afft::mpst::cpu::ExecutionParameters> || 
                         std::is_same_v<std::remove_cv_t<T>, afft::mpst::gpu::ExecutionParameters>> {};
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
