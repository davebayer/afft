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

#include <complex>
#include <type_traits>

#include "detail/type.hpp"

#if AFFT_GPU_FRAMEWORK_IS_CUDA
# if __has_include(<cuComplex.h>)
#   include <cuComplex.h>
# endif
# if __has_include(<cuda/std/complex>)
#   include <cuda/std/complex>
# endif
# if defined(AFFT_HAS_F16) && __has_include(<cuda_fp16.h>)
#   include <cuda_fp16.h>
# endif
# if defined(AFFT_HAS_BF16) && __has_include(<cuda_bf16.h>)
#   include <cuda_bf16.h>
# endif
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# if __has_include(<hip/hip_complex.h>)
#   include <hip/hip_complex.h>
# endif
# if defined(AFFT_HAS_F16) && __has_include(<hip_fp16.h>)
#   include <hip/hip_fp16.h>
# endif
# if defined(AFFT_HAS_BF16) && __has_include(<hip_bf16.h>)
#   include <hip/hip_bf16.h>
# endif
#endif

namespace afft
{
  /**
   * @brief Real data type.
   * @tparam prec The precision.
   */
  template<Precision prec>
    requires (detail::isValidPrecision(prec))
  using Real = detail::Float<prec>;

  /**
   * @brief Complex data type.
   * @tparam T The real type.
   */
  template<typename T>
  using Complex = std::complex<T>;

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
    // Ensure the size of the type matches the given precision and complexity
    static_assert(sizeof(T) == detail::sizeOf<prec, cmpl>(),
                  "Size of the type must match the given precision and complexity");

    // Ensure the precision is valid
    static_assert(detail::isValidPrecision(prec), "Invalid precision");

    // Ensure the complexity is valid
    static_assert(detail::isValidComplexity(cmpl), "Invalid complexity");

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
    : TypePropertiesBase<float, Precision::f32, Complexity::real> {};

  /// Specialization of TypeProperties for Complex<float> (std::complex<float>).
  template<>
  struct TypeProperties<Complex<float>>
    : TypePropertiesBase<Complex<float>, Precision::f32, Complexity::complex> {};

  /// Specialization of TypeProperties for double.
  template<>
  struct TypeProperties<double>
    : TypePropertiesBase<double, Precision::f64, Complexity::real> {};

  /// Specialization of TypeProperties for Complex<double> (std::complex<double>).
  template<>
  struct TypeProperties<Complex<double>>
    : TypePropertiesBase<Complex<double>, Precision::f64, Complexity::complex> {};

#if AFFT_GPU_FRAMEWORK_IS_CUDA
# if __has_include(<cuComplex.h>)
  /// Specialization of TypeProperties for cuFloatComplex.
  template<>
  struct TypeProperties<cuFloatComplex>
    : TypePropertiesBase<cuFloatComplex, Precision::f32, Complexity::complex> {};

  /// Specialization of TypeProperties for cuDoubleComplex.
  template<>
  struct TypeProperties<cuDoubleComplex>
    : TypePropertiesBase<cuDoubleComplex, Precision::f64, Complexity::complex> {};
# endif
# if __has_include(<cuda/std/complex>)
  /// Specialization of TypeProperties for cuda::std::complex<float>.
  template<>
  struct TypeProperties<cuda::std::complex<float>>
    : TypePropertiesBase<cuda::std::complex<float>, Precision::f32, Complexity::complex> {};

  /// Specialization of TypeProperties for cuda::std::complex<double>.
  template<>
  struct TypeProperties<cuda::std::complex<double>>
    : TypePropertiesBase<cuda::std::complex<double>, Precision::f64, Complexity::complex> {};
# endif
# if defined(AFFT_HAS_F16) && __has_include(<cuda_fp16.h>)
  /// Specialization of TypeProperties for half.
  template<>
  struct TypeProperties<half>
    : TypePropertiesBase<half, Precision::f16, Complexity::real> {};

  /// Specialization of TypeProperties for half2.
  template<>
  struct TypeProperties<half2>
    : TypePropertiesBase<half2, Precision::f16, Complexity::complex> {};
# endif
# if defined(AFFT_HAS_BF16) && __has_include(<cuda_bf16.h>)
  /// Specialization of TypeProperties for __nv_bfloat16.
  template<>
  struct TypeProperties<__nv_bfloat16>
    : TypePropertiesBase<__nv_bfloat16, Precision::bf16, Complexity::real> {};

  /// Specialization of TypeProperties for __nv_bfloat162.
  template<>
  struct TypeProperties<__nv_bfloat162>
    : TypePropertiesBase<__nv_bfloat162, Precision::bf16, Complexity::complex> {};
# endif
#elif AFFT_GPU_FRAMEWORK_IS_HIP
# if __has_include(<hip/hip_complex.h>)
  /// Specialization of TypeProperties for hipFloatComplex.
  template<>
  struct TypeProperties<hipFloatComplex>
    : TypePropertiesBase<hipFloatComplex, Precision::f32, Complexity::complex> {};

  /// Specialization of TypeProperties for hipDoubleComplex.
  template<>
  struct TypeProperties<hipDoubleComplex>
    : TypePropertiesBase<hipDoubleComplex, Precision::f64, Complexity::complex> {};
# endif
# if defined(AFFT_HAS_F16) && __has_include(<hip_fp16.h>)
  /// Specialization of TypeProperties for half.
  template<>
  struct TypeProperties<half>
    : TypePropertiesBase<half, Precision::f16, Complexity::real> {};

  /// Specialization of TypeProperties for half2.
  template<>
  struct TypeProperties<half2>
    : TypePropertiesBase<half2, Precision::f16, Complexity::complex> {};
# endif
# if defined(AFFT_HAS_BF16) && __has_include(<hip_bf16.h>)
  /// Specialization of TypeProperties for __hip_bfloat16.
  template<>
  struct TypeProperties<__hip_bfloat16>
    : TypePropertiesBase<__hip_bfloat16, Precision::bf16, Complexity::real> {};

  /// Specialization of TypeProperties for __hip_bfloat162.
  template<>
  struct TypeProperties<__hip_bfloat162>
    : TypePropertiesBase<__hip_bfloat162, Precision::bf16, Complexity::complex> {};
# endif
#endif

} // namespace afft

#endif /* AFFT_TYPE_HPP */
