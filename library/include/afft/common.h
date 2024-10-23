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

/// @brief Order type enumeration
enum afft_Order
{
  afft_Order_rowMajor,    ///< Row-major order
  afft_Order_columnMajor, ///< Column-major order
};

/// @brief Axis type
typedef uint8_t afft_Axis;

/// @brief Size type
typedef uint64_t afft_Size;

/// @brief Stride type
typedef size_t afft_Stride;

/// @brief FFTW3 library type enumeration
enum afft_fftw3_Library
{
  afft_fftw3_Library_float,      ///< FFTW3 single precision (fftwf)
  afft_fftw3_Library_double,     ///< FFTW3 double precision (fftw)
  afft_fftw3_Library_longDouble, ///< FFTW3 long double precision (fftwl)
  afft_fftw3_Library_quad,       ///< FFTW3 quadruple precision (fftwq)
};

/// @brief FFTW3 library type
typedef enum afft_fftw3_Library afft_fftw3_Library;

#ifdef __cplusplus
}
#endif

#endif /* AFFT_COMMON_H */
