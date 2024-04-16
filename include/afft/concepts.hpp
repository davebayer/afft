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

#ifndef AFFT_CONCEPTS_HPP
#define AFFT_CONCEPTS_HPP

#include <concepts>
#include <type_traits>

#include "typeTraits.hpp"
#include "detail/type.hpp"

namespace afft
{
  /**
   * @brief Known type concept.
   * @tparam T The type.
   */
  template<typename T>
  concept KnownType = !std::derived_from<TypeProperties<std::remove_cv_t<T>>, detail::UnknownTypePropertiesBase>
                      && std::derived_from<TypeProperties<std::remove_cv_t<T>>, detail::KnownTypePropertiesBase>
                      && detail::isValidPrecision(typePrecision<T>)
                      && detail::isValidComplexity(typeComplexity<T>);

  /**
   * @brief Real type concept.
   * @tparam T The type.
   */
  template<typename T>
  concept RealType = KnownType<T> && typeComplexity<T> == Complexity::real;

  /**
   * @brief Complex type concept.
   * @tparam T The type.
   */
  template<typename T>
  concept ComplexType = KnownType<T> && typeComplexity<T> == Complexity::complex;

  /**
   * @brief TransformParameters concept.
   * @tparam T The type.
   */
  template<typename T>
  concept TransformParametersType = isTransformParameters<T>;

  /**
   * @brief TargetParameters concept.
   * @tparam T The type.
   */
  template<typename T>
  concept TargetParametersType = isTargetParameters<T>;
} // namespace afft

#endif /* AFFT_CONCEPTS_HPP */
