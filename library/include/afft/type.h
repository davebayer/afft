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

#ifndef AFFT_TYPE_H
#define AFFT_TYPE_H

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
#define afft_Precision_bf16         (afft_Precision)0     ///< Google Brain's brain floating-point format
#define afft_Precision_f16          (afft_Precision)1     ///< IEEE 754 half-precision binary floating-point format
#define afft_Precision_f32          (afft_Precision)2     ///< IEEE 754 single-precision binary floating-point format
#define afft_Precision_f64          (afft_Precision)3     ///< IEEE 754 double-precision binary floating-point format
#define afft_Precision_f80          (afft_Precision)4     ///< x86 80-bit extended precision format
#define afft_Precision_f64f64       (afft_Precision)5     ///< double double precision (f128 simulated with two f64)
#define afft_Precision_f128         (afft_Precision)6     ///< IEEE 754 quadruple-precision binary floating-point format
#define afft_Precision_float        afft_Precision_f32    ///< Precision of float
#define afft_Precision_double       afft_Precision_f64    ///< Precision of double
#if LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024 && LDBL_MIN_EXP == -1021
# define afft_Precision_longDouble  afft_Precision_f64    ///< Precision of long double
#elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
# define afft_Precision_longDouble  afft_Precision_f80    ///< Precision of long double
#elif (LDBL_MANT_DIG >=   105 && LDBL_MANT_DIG <=   107) && \
      (LDBL_MAX_EXP  >=  1023 && LDBL_MAX_EXP  <=  1025) && \
      (LDBL_MIN_EXP  >= -1022 && LDBL_MIN_EXP  <= -1020)
# define afft_Precision_longDouble  afft_Precision_f64f64 ///< Precision of long double
#elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384 && LDBL_MIN_EXP == -16381
# define afft_Precision_longDouble  afft_Precision_f128   ///< Precision of long double
#else
# error "Unrecognized long double format"
#endif
#define afft_Precision_doubleDouble afft_Precision_f64f64 ///< Precision of double double
#define afft_Precision_quad         afft_Precision_f128   ///< Precision of quad

/// @brief Complexity type
typedef uint8_t afft_Complexity;

/// @brief Complexity enumeration
#define afft_Complexity_real    (afft_Complexity)0 ///< Real
#define afft_Complexity_complex (afft_Complexity)1 ///< Complex

#ifdef __cplusplus
}
#endif

#endif /* AFFT_TYPE_H */
