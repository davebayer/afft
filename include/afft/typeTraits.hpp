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

#ifndef AFFT_TYPE_TRAITS_HPP
#define AFFT_TYPE_TRAITS_HPP

#include <type_traits>

#include "detail/typeTraits.hpp"

namespace afft
{
  /**
   * @brief Get the precision of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Precision typePrecision = TypeProperties<std::remove_cv_t<T>>::precision;

  /**
   * @brief Get the complexity of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Complexity typeComplexity = TypeProperties<std::remove_cv_t<T>>::complexity;

  /**
   * @brief Target Parameters type for given transform.
   * @tparam transform The transform type.
   */
  template<Transform transform>
  using TransformParameters = typename detail::TransformParametersSelect<transform>::Type;

  /**
   * @brief Check if the type is TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTransformParameters = detail::IsTransformParameters<std::remove_cvref_t<T>>::value;

  /**
   * @brief Target Parameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  using TargetParameters = typename detail::TargetParametersSelect<target>::Type;

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTargetParameters = detail::IsTargetParameters<std::remove_cvref_t<T>>::value;

  /**
   * @brief Backend type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  using TargetBackend = typename detail::TargetBackendSelect<target>::Type;

  /**
   * @brief Check if the type is target transform backend.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTargetBackend = detail::IsTargetBackend<std::remove_cvref_t<T>>::value;

  /**
   * @brief BackendSelectParameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  using BackendSelectParameters = typename detail::TargetBackendSelectParametersSelect<target>::Type;

  /**
   * @brief Check if the type is BackendSelectParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isBackendSelectParameters = detail::IsBackendSelectParameters<std::remove_cvref_t<T>>::value;
  
  /**
   * @brief Get the target of the backend select parameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Target backendSelectParametersTarget = detail::BackendSelectParametersTarget<std::remove_cvref_t<T>>::value;
  
  /**
   * @brief Get the target of the target parameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Target targetParametersTarget = detail::TargetParametersTarget<std::remove_cvref_t<T>>::value;
} // namespace afft

#endif /* AFFT_TYPE_TRAITS_HPP */
