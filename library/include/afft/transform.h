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
#define afft_Transform_dft (afft_Transform)0 ///< Discrete Fourier Transform
#define afft_Transform_dht (afft_Transform)1 ///< Discrete Hartley Transform
#define afft_Transform_dtt (afft_Transform)2 ///< Discrete Trigonometric Transform

/// @brief Direction type
typedef uint8_t afft_Direction;

/// @brief Direction enumeration
#define afft_Direction_forward (afft_Direction)0       ///< Forward
#define afft_Direction_inverse (afft_Direction)1       ///< Inverse
#define afft_Direction_backward afft_Direction_inverse ///< Alias for inverse

/// @brief Normalization type
typedef uint8_t afft_Normalization;

/// @brief Normalization enumeration
#define afft_Normalization_none       (afft_Normalization)0 ///< No normalization
#define afft_Normalization_orthogonal (afft_Normalization)1 ///< 1/sqrt(N) normalization applied to both forward and inverse transform
#define afft_Normalization_unitary    (afft_Normalization)2 ///< 1/N normalization applied to inverse transform

/// @brief Placement type
typedef uint8_t afft_Placement;

/// @brief Placement enumeration
#define afft_Placement_outOfPlace (afft_Placement)0         ///< Out-of-place
#define afft_Placement_inPlace    (afft_Placement)1         ///< In-place
#define afft_Placement_notInPlace afft_Placement_outOfPlace ///< Alias for outOfPlace

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
#define afft_dft_Type_complexToComplex (afft_dft_Type)0               ///< Complex-to-complex transform
#define afft_dft_Type_realToComplex    (afft_dft_Type)1               ///< Real-to-complex transform
#define afft_dft_Type_complexToReal    (afft_dft_Type)2               ///< Complex-to-real transform
#define afft_dft_Type_c2c              afft_dft_Type_complexToComplex ///< Alias for complex-to-complex transform
#define afft_dft_Type_r2c              afft_dft_Type_realToComplex    ///< Alias for real-to-complex transform
#define afft_dft_Type_c2r              afft_dft_Type_complexToReal    ///< Alias for complex-to-real transform

/// @brief DFT parameters structure
struct afft_dft_Parameters
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const afft_Size*    shape;         ///< Shape of the transform
  size_t              transformRank; ///< Rank of the transform
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
#define afft_dht_Type_separable (afft_dht_Type)0 ///< Separable DHT, computes the DHT along each axis independently

/// @brief DHT parameters structure
struct afft_dht_Parameters
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const afft_Size*    shape;         ///< Shape of the transform
  size_t              transformRank; ///< Rank of the transform
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
#define afft_dtt_Type_dct1 (afft_dtt_Type)0   ///< Discrete Cosine Transform type I
#define afft_dtt_Type_dct2 (afft_dtt_Type)1   ///< Discrete Cosine Transform type II
#define afft_dtt_Type_dct3 (afft_dtt_Type)2   ///< Discrete Cosine Transform type III
#define afft_dtt_Type_dct4 (afft_dtt_Type)3   ///< Discrete Cosine Transform type IV
#define afft_dtt_Type_dct5 (afft_dtt_Type)4   ///< Discrete Cosine Transform type V
#define afft_dtt_Type_dct6 (afft_dtt_Type)5   ///< Discrete Cosine Transform type VI
#define afft_dtt_Type_dct7 (afft_dtt_Type)6   ///< Discrete Cosine Transform type VII
#define afft_dtt_Type_dct8 (afft_dtt_Type)7   ///< Discrete Cosine Transform type VIII
#define afft_dtt_Type_dst1 (afft_dtt_Type)8   ///< Discrete Sine Transform type I
#define afft_dtt_Type_dst2 (afft_dtt_Type)9   ///< Discrete Sine Transform type II
#define afft_dtt_Type_dst3 (afft_dtt_Type)10  ///< Discrete Sine Transform type III
#define afft_dtt_Type_dst4 (afft_dtt_Type)11  ///< Discrete Sine Transform type IV
#define afft_dtt_Type_dst5 (afft_dtt_Type)12  ///< Discrete Sine Transform type V
#define afft_dtt_Type_dst6 (afft_dtt_Type)13  ///< Discrete Sine Transform type VI
#define afft_dtt_Type_dst7 (afft_dtt_Type)14  ///< Discrete Sine Transform type VII
#define afft_dtt_Type_dst8 (afft_dtt_Type)15  ///< Discrete Sine Transform type VIII
#define afft_dtt_Type_dct  afft_dtt_Type_dct2 ///< alias for Discrete Cosine Transform type II
#define afft_dtt_Type_dst  afft_dtt_Type_dst2 ///< alias for Discrete Sine Transform type II

/// @brief DTT parameters structure
struct afft_dtt_Parameters
{
  afft_Direction       direction;     ///< Direction of the transform
  afft_PrecisionTriad  precision;     ///< Precision triad
  size_t               shapeRank;     ///< Rank of the shape
  const afft_Size*     shape;         ///< Shape of the transform
  size_t               transformRank; ///< Rank of the transform
  const afft_Axis*     axes;          ///< Axes of the transform
  afft_Normalization   normalization; ///< Normalization
  afft_Placement       placement;     ///< Placement of the transform
  const afft_dtt_Type* types;         ///< Types of the transform
};

/**********************************************************************************************************************/
// Transform parameters variant
/**********************************************************************************************************************/
/// @brief Transform parameters variant
typedef struct afft_TransformParametersVariant afft_TransformParametersVariant;

/// @brief Transform parameters variant structure
struct afft_TransformParametersVariant
{
  afft_Transform transform;  ///< Transform type
  union
  {
    afft_dft_Parameters dft; ///< DFT parameters
    afft_dht_Parameters dht; ///< DHT parameters
    afft_dtt_Parameters dtt; ///< DTT parameters
  };
};

#ifdef __cplusplus
}
#endif

#endif /* AFFT_TRANSFORM_H */
