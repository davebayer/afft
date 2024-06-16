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
#ifndef AFFT_TRANSFORM_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**********************************************************************************************************************/
// Discrete Fourier Transform (DFT)
/**********************************************************************************************************************/
/// @brief DFT transform type
typedef uint8_t afft_dft_Type;

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

/// @brief DFT parameters enumeration
typedef struct
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const size_t*       shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const size_t*       axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dft_Type       type;          ///< Type of the transform
} afft_dft_Parameters;

/**********************************************************************************************************************/
// Discrete Hartley Transform (DHT)
/**********************************************************************************************************************/
/// @brief DHT transform type
typedef uint8_t afft_dht_Type;

/// @brief DHT transform enumeration
enum
{
  afft_dht_Type_separable, ///< Separable DHT, computes the DHT along each axis independently
};

/// @brief DHT parameters structure
typedef struct
{
  afft_Direction      direction;     ///< Direction of the transform
  afft_PrecisionTriad precision;     ///< Precision triad
  size_t              shapeRank;     ///< Rank of the shape
  const size_t*       shape;         ///< Shape of the transform
  size_t              axesRank;      ///< Rank of the axes
  const size_t*       axes;          ///< Axes of the transform
  afft_Normalization  normalization; ///< Normalization
  afft_Placement      placement;     ///< Placement of the transform
  afft_dht_Type       type;          ///< Type of the transform
} afft_dht_Parameters;

/**********************************************************************************************************************/
// Discrete Trigonometric Transform (DTT)
/**********************************************************************************************************************/
/// @brief DTT transform type
typedef uint8_t afft_dtt_Type;

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
typedef struct
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
} afft_dtt_Parameters;

/**********************************************************************************************************************/
// General transform parameters
/**********************************************************************************************************************/
/// @brief Transform parameters structure
typedef struct
{
  union
  {
    afft_dft_Parameters dft;
    afft_dht_Parameters dht;
    afft_dtt_Parameters dtt;
  };
  afft_Transform        transform;
} afft_TransformParameters;

/**********************************************************************************************************************/
// Private functions
/**********************************************************************************************************************/
static inline afft_TransformParameters _afft_makeTransformParametersDft(afft_dft_Parameters params)
{
  afft_TransformParameters result;
  result.dft       = params;
  result.transform = afft_Transform_dft;

  return result;
}

static inline afft_TransformParameters _afft_makeTransformParametersDht(afft_dht_Parameters params)
{
  afft_TransformParameters result;
  result.dht       = params;
  result.transform = afft_Transform_dht;

  return result;
}

static inline afft_TransformParameters _afft_makeTransformParametersDtt(afft_dtt_Parameters params)
{
  afft_TransformParameters result;
  result.dtt       = params;
  result.transform = afft_Transform_dtt;

  return result;
}

static inline afft_TransformParameters _afft_makeTransformParametersAny(afft_TransformParameters params)
{
  return params;
}

/**********************************************************************************************************************/
// Public functions
/**********************************************************************************************************************/
#ifdef __cplusplus
} // extern "C"

/**
 * @brief Make transform parameters
 * @param params DFT parameters
 * @return Transform parameters
 */
static inline afft_TransformParameters afft_makeTransformParameters(afft_dft_Parameters params)
{
  return _afft_makeTransformParametersDft(params);
}

/**
 * @brief Make transform parameters
 * @param params DHT parameters
 * @return Transform parameters
 */
static inline afft_TransformParameters afft_makeTransformParameters(afft_dht_Parameters params)
{
  return _afft_makeTransformParametersDht(params);
}

/**
 * @brief Make transform parameters
 * @param params DTT parameters
 * @return Transform parameters
 */
static inline afft_TransformParameters afft_makeTransformParameters(afft_dtt_Parameters params)
{
  return _afft_makeTransformParametersDtt(params);
}

/**
 * @brief Make transform parameters
 * @param params Transform parameters
 * @return Transform parameters
 */
static inline afft_TransformParameters afft_makeTransformParameters(afft_TransformParameters params)
{
  return _afft_makeTransformParametersAny(params);
}

extern "C"
{
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  /**
   * @brief Make transform parameters
   * @param params DFT, DHT, DTT or general parameters
   * @return Transform parameters
   */
# define afft_makeTransformParameters(params) _Generic((params), \
    afft_dft_Parameters:      _afft_makeTransformParametersDft, \
    afft_dht_Parameters:      _afft_makeTransformParametersDht, \
    afft_dtt_Parameters:      _afft_makeTransformParametersDtt, \
    afft_TransformParameters: _afft_makeTransformParametersAny)(params)
#endif

#ifdef __cplusplus
}
#endif

#endif /* AFFT_TRANSFORM_H */
