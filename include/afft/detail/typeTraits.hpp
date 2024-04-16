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

namespace afft::detail
{
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
  struct TargetTransformBackendSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetTransformBackendSelect<Target::cpu>
  {
    using Type = afft::cpu::TransformBackend;
  };

  /// @brief Specialization for gpu target.
  template<>
  struct TargetTransformBackendSelect<Target::gpu>
  {
    using Type = afft::gpu::TransformBackend;
  };

  /**
   * @brief Check if the type is target transform backend.
   * @tparam T The type.
   */
  template<typename>
  struct IsTargetTransformBackend : std::false_type {};

  /// @brief Specialization for cpu transform backend.
  template<>
  struct IsTargetTransformBackend<afft::cpu::TransformBackend> : std::true_type {};

  /// @brief Specialization for gpu transform backend.
  template<>
  struct IsTargetTransformBackend<afft::gpu::TransformBackend> : std::true_type {};
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
