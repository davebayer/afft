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

#ifndef AFFT_TYPE_HPP
#define AFFT_TYPE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "detail/type.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @brief Planar complex data type.
   * @tparam T The real type.
   */
  template<typename T>
  struct PlanarComplex
  {
    T real{}; ///< The real part.
    T imag{}; ///< The imaginary part.
  };

  /**
   * @brief Deduction guide for PlanarComplex.
   * @tparam T The real type.
   */
  template<typename T>
  PlanarComplex(T r, T i) -> PlanarComplex<T>;

  /**
   * @struct TypePropertiesBase
   * @brief Type properties base structure implementing TypeProperties's static members. Should not be used directly,
   *        only as the base class for the TypeProperties specialization.
   * @tparam T The type.
   * @tparam prec The precision.
   * @tparam cmpl The complexity.
   */
  template<typename T, Precision prec, Complexity cmpl>
  struct TypePropertiesBase : detail::KnownTypePropertiesBase
  {
    // Check if type is an object
    static_assert(std::is_object_v<T>, "Type T must be an object");

    // Check if type is not abstract
    static_assert(!std::is_abstract_v<T>, "Type T cannot be abstract");

    // Ensure the size of the type matches the given precision and complexity
    static_assert(sizeof(T) == detail::sizeOf<prec, cmpl>(),
                  "Size of the type must match the given precision and complexity");

    // Ensure the precision is valid
    static_assert(detail::isValid(prec), "Invalid precision");

    // Ensure the complexity is valid
    static_assert(detail::isValid(cmpl), "Invalid complexity");

    static constexpr Precision  precision{prec};  ///< The precision.
    static constexpr Complexity complexity{cmpl}; ///< The complexity.
  };

  /**
   * @struct TypeProperties
   * @brief Type properties structure for the given type. Specialize this structure for new types.
   * @tparam T The type.
   */
  template<typename>
  struct TypeProperties
    : detail::UnknownTypePropertiesBase {};

  /// Specialization of TypeProperties for float.
  template<>
  struct TypeProperties<float>
    : TypePropertiesBase<float, Precision::_float, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<float>.
  template<>
  struct TypeProperties<std::complex<float>>
    : TypePropertiesBase<std::complex<float>, Precision::_float, Complexity::complex> {};

  /// Specialization of TypeProperties for double.
  template<>
  struct TypeProperties<double>
    : TypePropertiesBase<double, Precision::_double, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<double>.
  template<>
  struct TypeProperties<std::complex<double>>
    : TypePropertiesBase<std::complex<double>, Precision::_double, Complexity::complex> {};

  /// Specialization of TypeProperties for long double.
  template<>
  struct TypeProperties<long double>
    : TypePropertiesBase<long double, Precision::_longDouble, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<long double>.
  template<>
  struct TypeProperties<std::complex<long double>>
    : TypePropertiesBase<std::complex<long double>, Precision::_longDouble, Complexity::complex> {};

#ifdef AFFT_CXX_HAS_STD_FLOAT
# ifdef __STDCPP_BFLOAT16_T__
  /// Specialization of TypeProperties for std::bfloat16_t.
  template<>
  struct TypeProperties<std::bfloat16_t>
    : TypePropertiesBase<std::bfloat16_t, Precision::bf16, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<std::bfloat16_t>.
  template<>
  struct TypeProperties<std::complex<std::bfloat16_t>>
    : TypePropertiesBase<std::complex<std::bfloat16_t>, Precision::bf16, Complexity::complex> {}; 
#endif
# ifdef __STDCPP_FLOAT16_T__
  /// Specialization of TypeProperties for std::float16_t.
  template<>
  struct TypeProperties<std::float16_t>
    : TypePropertiesBase<std::float16_t, Precision::f16, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<std::float16_t>.
  template<>
  struct TypeProperties<std::complex<std::float16_t>>
    : TypePropertiesBase<std::complex<std::float16_t>, Precision::f16, Complexity::complex> {};
# endif
# ifdef __STDCPP_FLOAT32_T__
  /// Specialization of TypeProperties for std::float32_t.
  template<>
  struct TypeProperties<std::float32_t>
    : TypePropertiesBase<std::float32_t, Precision::f32, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<std::float32_t>.
  template<>
  struct TypeProperties<std::complex<std::float32_t>>
    : TypePropertiesBase<std::complex<std::float32_t>, Precision::f32, Complexity::complex> {};
# endif
# ifdef __STDCPP_FLOAT64_T__
  /// Specialization of TypeProperties for std::float64_t.
  template<>
  struct TypeProperties<std::float64_t>
    : TypePropertiesBase<std::float64_t, Precision::f64, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<std::float64_t>.
  template<>
  struct TypeProperties<std::complex<std::float64_t>>
    : TypePropertiesBase<std::complex<std::float64_t>, Precision::f64, Complexity::complex> {};
# endif
# ifdef __STDCPP_FLOAT128_T__
  /// Specialization of TypeProperties for std::float128_t.
  template<>
  struct TypeProperties<std::float128_t>
    : TypePropertiesBase<std::float128_t, Precision::f128, Complexity::real> {};

  /// Specialization of TypeProperties for std::complex<std::float128_t>.
  template<>
  struct TypeProperties<std::complex<std::float128_t>>
    : TypePropertiesBase<std::complex<std::float128_t>, Precision::f128, Complexity::complex> {};
# endif
#endif
} // namespace afft

#endif /* AFFT_TYPE_HPP */
