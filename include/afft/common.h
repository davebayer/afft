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

#ifndef AFFT_COMMON_H
#define AFFT_COMMON_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Precision type
typedef uint8_t afft_Precision;

/// @brief Precision enumeration
enum
{
  afft_Precision_bf16,   ///< Google Brain's brain floating-point format
  afft_Precision_f16,    ///< IEEE 754 half-precision binary floating-point format
  afft_Precision_f32,    ///< IEEE 754 single-precision binary floating-point format
  afft_Precision_f64,    ///< IEEE 754 double-precision binary floating-point format
  afft_Precision_f80,    ///< x86 80-bit extended precision format
  afft_Precision_f64f64, ///< double double precision (f128 simulated with two f64)
  afft_Precision_f128,   ///< IEEE 754 quadruple-precision binary floating-point format

  afft_Precision_float        = afft_Precision_f32,    ///< Precision of float
  afft_Precision_double       = afft_Precision_f64,    ///< Precision of double
# if LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024 && LDBL_MIN_EXP == -1021
  afft_Precision_float        = afft_Precision_f64,    ///< Precision of long double
# elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
  afft_Precision_longDouble   = afft_Precision_f80,    ///< Precision of long double
# elif (LDBL_MANT_DIG >=   105 && LDBL_MANT_DIG <=   107) && \
       (LDBL_MAX_EXP  >=  1023 && LDBL_MAX_EXP  <=  1025) && \
       (LDBL_MIN_EXP  >= -1022 && LDBL_MIN_EXP  <= -1020)
  afft_Precision_longDouble   = afft_Precision_f64f64, ///< Precision of long double
# elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
  afft_Precision_longDouble   = afft_Precision_f128,   ///< Precision of long double
# else
#   error "Unrecognized long double format"
# endif
  afft_Precision_doubleDouble = afft_Precision_f64f64, ///< Precision of double double
  afft_Precision_quad         = afft_Precision_f128,   ///< Precision of quad
};

/// @brief Alignment type
typedef size_t afft_Alignment;

/// @brief Alignment enumeration
enum
{
  afft_Alignment_simd128  = 16,  ///< 128-bit SIMD alignment
  afft_Alignment_simd256  = 32,  ///< 256-bit SIMD alignment
  afft_Alignment_simd512  = 64,  ///< 512-bit SIMD alignment
  afft_Alignment_simd1024 = 128, ///< 1024-bit SIMD alignment
  afft_Alignment_simd2048 = 256, ///< 2048-bit SIMD alignment

  afft_Alignment_sse    = afft_Alignment_simd128,  ///< SSE alignment
  afft_Alignment_sse2   = afft_Alignment_simd128,  ///< SSE2 alignment
  afft_Alignment_sse3   = afft_Alignment_simd128,  ///< SSE3 alignment
  afft_Alignment_sse4   = afft_Alignment_simd128,  ///< SSE4 alignment
  afft_Alignment_sse4_1 = afft_Alignment_simd128,  ///< SSE4.1 alignment
  afft_Alignment_sse4_2 = afft_Alignment_simd128,  ///< SSE4.2 alignment
  afft_Alignment_avx    = afft_Alignment_simd256,  ///< AVX alignment
  afft_Alignment_avx2   = afft_Alignment_simd256,  ///< AVX2 alignment
  afft_Alignment_avx512 = afft_Alignment_simd512,  ///< AVX-512 alignment
  afft_Alignment_neon   = afft_Alignment_simd128,  ///< NEON alignment
  afft_Alignment_sve    = afft_Alignment_simd2048, ///< SVE alignment
};

/// @brief Complexity type
typedef uint8_t afft_Complexity;

/// @brief Complexity enumeration
enum
{
  afft_Complexity_real,    ///< Real
  afft_Complexity_complex, ///< Complex
};

/// @brief Complex format type
typedef uint8_t afft_ComplexFormat;

/// @brief Complex format enumeration
enum
{
  afft_ComplexFormat_interleaved, ///< Interleaved
  afft_ComplexFormat_planar       ///< Planar
};

/// @brief Direction type
typedef uint8_t afft_Direction;

/// @brief Direction enumeration
enum
{
  afft_Direction_forward, ///< Forward
  afft_Direction_inverse, ///< Inverse

  afft_Direction_backward = afft_Direction_inverse, ///< Alias for inverse
};

/// @brief Placement type
typedef uint8_t afft_Placement;

/// @brief Placement enumeration
enum
{
  afft_Placement_inPlace,    ///< In-place
  afft_Placement_outOfPlace, ///< Out-of-place

  afft_Placement_notInPlace = afft_Placement_outOfPlace, ///< Alias for outOfPlace
};

/// @brief Transform type
typedef uint8_t afft_Transform;

/// @brief Transform enumeration
enum
{
  afft_Transform_dft, ///< Discrete Fourier Transform
  afft_Transform_dht, ///< Discrete Hartley Transform
  afft_Transform_dtt, ///< Discrete Trigonometric Transform
};

/// @brief Target type
typedef uint8_t afft_Target;

/// @brief Target enumeration
enum
{
  afft_Target_cpu, ///< CPU
  afft_Target_gpu, ///< GPU
};

/// @brief Distribution type
typedef uint8_t afft_Distribution;

/// @brief Distribution enumeration
enum
{
  afft_Distribution_spst, ///< Single process, single target
  afft_Distribution_spmt, ///< Single process, multiple targets
  afft_Distribution_mpst, ///< Multiple processes, single target

  afft_Distribution_single = afft_Distribution_spst, ///< Alias for single process, single target
  afft_Distribution_multi  = afft_Distribution_spmt, ///< Alias for single process, multiple targets
  afft_Distribution_mpi    = afft_Distribution_mpst, ///< Alias for multiple processes, single target
};

/// @brief Normalization type
typedef uint8_t afft_Normalization;

/// @brief Normalization enumeration
enum
{
  afft_Normalization_none,       ///< No normalization
  afft_Normalization_orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
  afft_Normalization_unitary,    ///< 1/N normalization applied to inverse transform
};

/// @brief Precision triad structure
typedef struct
{
  afft_Precision execution;   ///< Precision of the execution
  afft_Precision source;      ///< Precision of the source data
  afft_Precision destination; ///< Precision of the destination data
} afft_PrecisionTriad;

#ifdef __cplusplus
}
#endif

#endif /* AFFT_COMMON_H */
