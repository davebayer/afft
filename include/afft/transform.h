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

#ifndef AFFT_TRANSFORM_H
#define AFFT_TRANSFORM_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"
#include "type.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Transform type
typedef uint8_t afft_Transform;

/// @brief Transform enumeration
enum
{
  afft_Transform_dft, ///< Discrete Fourier Transform
  afft_Transform_dht, ///< Discrete Hartley Transform
  afft_Transform_dtt, ///< Discrete Trigonometric Transform
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

/// @brief Normalization type
typedef uint8_t afft_Normalization;

/// @brief Normalization enumeration
enum
{
  afft_Normalization_none,       ///< No normalization
  afft_Normalization_orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
  afft_Normalization_unitary,    ///< 1/N normalization applied to inverse transform
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

/// @brief Precision triad type
typedef struct afft_PrecisionTriad afft_PrecisionTriad;

/// @brief Precision triad structure
struct afft_PrecisionTriad
{
  afft_Precision execution;   ///< Precision of the execution
  afft_Precision source;      ///< Precision of the source data
  afft_Precision destination; ///< Precision of the destination data
};

/// @brief Named constant representing all axes
#define AFFT_ALL_AXES ((afft_Axis*)0)

/**********************************************************************************************************************/
// Discrete Fourier Transform (DFT)
/**********************************************************************************************************************/
/// @brief DFT transform type
typedef uint8_t afft_dft_Type;

/// @brief DFT transform parameters structure
typedef struct afft_dft_Parameters afft_dft_Parameters;

/// @brief DFT transform enumeration
enum
{
  afft_dft_Type_complexToComplex, ///< Complex-to-complex transform
  afft_dft_Type_realToComplex,    ///< Real-to-complex transform
  afft_dft_Type_complexToReal,    ///< Complex-to-real transform

  afft_dft_Type_c2c = afft_dft_Type_complexToComplex, ///< Alias for complex-to-complex transform
  afft_dft_Type_r2c = afft_dft_Type_realToComplex,    ///< Alias for real-to-complex transform
  afft_dft_Type_c2r = afft_dft_Type_complexToReal,    ///< Alias for complex-to-real transform
};

/// @brief DFT parameters structure
struct afft_dft_Parameters
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const afft_Size*    shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const afft_Axis*    axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dft_Type       type;          ///< Type of the transform
};

/**********************************************************************************************************************/
// Discrete Hartley Transform (DHT)
/**********************************************************************************************************************/
/// @brief DHT transform type
typedef uint8_t afft_dht_Type;

/// @brief DHT transform parameters structure
typedef struct afft_dht_Parameters afft_dht_Parameters;

/// @brief DHT transform enumeration
enum
{
  afft_dht_Type_separable, ///< Separable DHT, computes the DHT along each axis independently
};

/// @brief DHT parameters structure
struct afft_dht_Parameters
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const afft_Size*    shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const afft_Axis*    axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dht_Type       type;          ///< Type of the transform
};

/**********************************************************************************************************************/
// Discrete Trigonometric Transform (DTT)
/**********************************************************************************************************************/
/// @brief DTT transform type
typedef uint8_t afft_dtt_Type;

/// @brief DTT transform parameters structure
typedef struct afft_dtt_Parameters afft_dtt_Parameters;

/// @brief DTT transform enumeration
enum
{
  afft_dtt_Type_dct1, ///< Discrete Cosine Transform type I
  afft_dtt_Type_dct2, ///< Discrete Cosine Transform type II
  afft_dtt_Type_dct3, ///< Discrete Cosine Transform type III
  afft_dtt_Type_dct4, ///< Discrete Cosine Transform type IV

  afft_dtt_Type_dst1, ///< Discrete Sine Transform type I
  afft_dtt_Type_dst2, ///< Discrete Sine Transform type II
  afft_dtt_Type_dst3, ///< Discrete Sine Transform type III
  afft_dtt_Type_dst4, ///< Discrete Sine Transform type IV

  afft_dtt_Type_dct = afft_dtt_Type_dct2, ///< default DCT type
  afft_dtt_Type_dst = afft_dtt_Type_dst2, ///< default DST type
};

/// @brief DTT parameters structure
struct afft_dtt_Parameters
{
  afft_Direction       direction;     ///< Direction of the transform
  afft_PrecisionTriad  precision;     ///< Precision triad
  size_t               shapeRank;     ///< Rank of the shape
  const size_t*        shape;         ///< Shape of the transform
  size_t               axesRank;      ///< Rank of the axes
  const size_t*        axes;          ///< Axes of the transform
  afft_Normalization   normalization; ///< Normalization
  afft_Placement       placement;     ///< Placement of the transform
  const afft_dtt_Type* types;         ///< Types of the transform
};

#ifdef __cplusplus
}
#endif

#endif /* AFFT_TRANSFORM_H */
