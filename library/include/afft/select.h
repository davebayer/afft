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

#ifndef AFFT_SELECT_H
#define AFFT_SELECT_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "backend.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Select strategy type
typedef uint8_t afft_SelectStrategy;

/// @brief Select strategy enumeration
#define afft_SelectStrategy_first   (afft_SelectStrategy)0    ///< Select the first available backend
#define afft_SelectStrategy_best    (afft_SelectStrategy)1    ///< Select the best available backend
#define afft_SelectStrategy_default afft_SelectStrategy_first ///< Default select strategy, alias for first

/// @brief First backend selection parameters
typedef struct afft_FirstSelectParameters afft_FirstSelectParameters;

/// @brief Select parameters for selecting first backend supporting the transform
struct afft_FirstSelectParameters
{
  afft_BackendMask    mask;      ///< Backend mask
  size_t              orderSize; ///< Number of backends in the order
  const afft_Backend* order;     ///< Order of the backends
};

/// @brief Select parameters for selecting best of all the backends supporting the transform
typedef struct afft_BestSelectParameters afft_BestSelectParameters;

/// @brief Select parameters for selecting best of all the backends supporting the transform
struct afft_BestSelectParameters
{
  afft_BackendMask mask;                   ///< Backend mask
  double           destructiveTimePenalty; ///< Time penalty for destructive backends
};

/// @brief Default select parameters
typedef afft_FirstSelectParameters afft_DefaultSelectParameters;

#ifdef __cplusplus
}
#endif

#endif /* AFFT_SELECT_H */
