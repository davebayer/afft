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

#ifndef AFFT_COMMON_HPP
#define AFFT_COMMON_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "Span.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Maximum number of dimensions
  inline constexpr std::size_t maxDimCount{AFFT_MAX_DIM_COUNT};

  /// @brief Dynamic rank
  inline constexpr std::size_t dynamicRank{dynamicExtent};

  /// @brief Precision of a floating-point number
  enum class Precision : std::uint8_t
  {
    bf16,        ///< Google Brain's brain floating-point format
    f16,         ///< IEEE 754 half-precision binary floating-point format
    f32,         ///< IEEE 754 single-precision binary floating-point format
    f64,         ///< IEEE 754 double-precision binary floating-point format
    f80,         ///< x86 80-bit extended precision format
    f64f64,      ///< double double precision (f128 simulated with two f64)
    f128,        ///< IEEE 754 quadruple-precision binary floating-point format

    _float        = f32,    ///< Precision of float
    _double       = f64,    ///< Precision of double
# if LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024 && LDBL_MIN_EXP == -1021
    _longDouble   = f64,    ///< Precision of long double
# elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
    _longDouble   = f80,    ///< Precision of long double
# elif (LDBL_MANT_DIG >=   105 && LDBL_MANT_DIG <=   107) && \
       (LDBL_MAX_EXP  >=  1023 && LDBL_MAX_EXP  <=  1025) && \
       (LDBL_MIN_EXP  >= -1022 && LDBL_MIN_EXP  <= -1020)
    _longDouble   = f64f64, ///< Precision of long double
# elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
    _longDouble   = f128,   ///< Precision of long double
# else
#   error "Unrecognized long double format"
# endif
    _doubleDouble = f64f64, ///< Precision of double double
    _quad         = f128,   ///< Precision of quad
  };

  /// @brief Alignment of a data type
  enum class Alignment : std::size_t
  {
    defaultNew = __STDCPP_DEFAULT_NEW_ALIGNMENT__, ///< Default alignment for new operator
    simd128    = 16,                               ///< 128-bit SIMD alignment
    simd256    = 32,                               ///< 256-bit SIMD alignment
    simd512    = 64,                               ///< 512-bit SIMD alignment
    simd1024   = 128,                              ///< 1024-bit SIMD alignment
    simd2048   = 256,                              ///< 2048-bit SIMD alignment

    sse    = simd128,  ///< SSE alignment
    sse2   = simd128,  ///< SSE2 alignment
    sse3   = simd128,  ///< SSE3 alignment
    sse4   = simd128,  ///< SSE4 alignment
    sse4_1 = simd128,  ///< SSE4.1 alignment
    sse4_2 = simd128,  ///< SSE4.2 alignment
    avx    = simd256,  ///< AVX alignment
    avx2   = simd256,  ///< AVX2 alignment
    avx512 = simd512,  ///< AVX-512 alignment
    neon   = simd128,  ///< NEON alignment
    sve    = simd2048, ///< SVE alignment
  };

  /// @brief Complexity of a data type
  enum class Complexity : std::uint8_t
  {
    real,    ///< real
    complex, ///< complex
  };

  /// @brief Complex number format
  enum class ComplexFormat : std::uint8_t
  {
    interleaved, ///< interleaved complex format
    planar,      ///< planar complex format
  };

  /// @brief Direction of the transform
  enum class Direction : std::uint8_t
  {
    forward,            ///< forward transform
    inverse,            ///< inverse transform
    backward = inverse, ///< alias for inverse transform
  };

  /// @brief Placement of the transform
  enum class Placement : std::uint8_t
  {
    inPlace,                 ///< in-place transform
    outOfPlace,              ///< out-of-place transform
    notInPlace = outOfPlace, ///< alias for outOfPlace transform
  };

  /// @brief Transform type
  enum class Transform : std::uint8_t
  {
    dft, ///< Discrete Fourier Transform
    dht, ///< Discrete Hartley Transform
    dtt, ///< Discrete Trigonometric Transform
  };

  /// @brief Target
  enum class Target : std::uint8_t
  {
    cpu, ///< CPU target
    gpu, ///< GPU target
  };

  /// @brief Distribution type
  enum class Distribution : std::uint8_t
  {
    spst,           ///< single process, single target
    spmt,           ///< single process, multiple targets
    mpst,           ///< multiple processes, single target
    
    single = spst,  ///< alias for single process, single target
    multi  = spmt,  ///< alias for single process, multiple targets
    mpi    = mpst,  ///< alias for multiple processes, single target
  };

  /// @brief Normalization
  enum class Normalization : std::uint8_t
  {
    none,       ///< no normalization
    orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
    unitary,    ///< 1/N normalization applied to inverse transform
  };

  /**
   * @struct PrecisionTriad
   * @brief Precision triad
   */
  struct PrecisionTriad
  {
    Precision execution{};   ///< precision of the execution
    Precision source{};      ///< precision of the source data
    Precision destination{}; ///< precision of the destination data
  };

  /**
   * @brief Equality operator for PrecisionTriad
   * @param lhs left-hand side
   * @param rhs right-hand side
   * @return true if the precision triads are equal, false otherwise
   */
  [[nodiscard]] inline constexpr bool operator==(const PrecisionTriad& lhs, const PrecisionTriad& rhs)
  {
    return (lhs.execution == rhs.execution) &&
           (lhs.source == rhs.source) &&
           (lhs.destination == rhs.destination);
  }
} // namespace afft

#endif /* AFFT_COMMON_HPP */
