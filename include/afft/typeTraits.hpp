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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "type.hpp"
#include "detail/cxx.hpp"
#include "detail/typeTraits.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @brief Get the precision of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Precision typePrecision = detail::TypePropertiesHelper<std::remove_cv_t<T>>::precision;

  /**
   * @brief Get the complexity of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Complexity typeComplexity = detail::TypePropertiesHelper<std::remove_cv_t<T>>::complexity;

  /**
   * @brief Check if the type is a known type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isKnownType = std::is_base_of_v<detail::KnownTypePropertiesBase, detail::TypePropertiesHelper<std::remove_cv_t<T>>>;

  /**
   * @brief Check if the type is a known real type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isRealType = isKnownType<T> && typeComplexity<T> == Complexity::real;

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
  inline constexpr bool isTransformParameters = detail::IsTransformParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief Target Parameters type for given target.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib = Distribution::spst>
  using TargetParameters = typename detail::TargetParametersSelect<target, distrib>::Type;

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTargetParameters = detail::IsTargetParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief Select Parameters type for given target.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib = Distribution::spst>
  using SelectParameters = typename detail::SelectParametersSelect<target, distrib>::Type;

  /**
   * @brief Is the type SelectParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isSelectParameters = detail::IsSelectParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief ExecutionParameters type for given target.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target, Distribution distrib = Distribution::spst>
  using ExecutionParameters = typename detail::TargetExecutionParametersSelect<target, distrib>::Type;

  /**
   * @brief Check if the type is ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isExecutionParameters = detail::IsExecutionParameters<detail::cxx::remove_cvref_t<T>>::value;
} // namespace afft

#endif /* AFFT_TYPE_TRAITS_HPP */
