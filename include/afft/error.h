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

#ifndef AFFT_ERROR_H
#define AFFT_ERROR_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Error enumeration
typedef enum
{
  afft_Error_success,
  afft_Error_internal,
  afft_Error_invalidPlan,
  afft_Error_invalidArgument,

  afft_Error_invalidPrecision,
  afft_Error_invalidAlignment,
  afft_Error_invalidComplexity,
  afft_Error_invalidComplexFormat,
  afft_Error_invalidDirection,
  afft_Error_invalidPlacement,
  afft_Error_invalidTransform,
  afft_Error_invalidTarget,
  afft_Error_invalidDistribution,
  afft_Error_invalidNormalization,
  afft_Error_invalidBackend,
  afft_Error_invalidSelectStrategy,
  afft_Error_invalidCufftWorkspacePolicy,
  afft_Error_invalidFftw3PlannerFlag,
  afft_Error_invalidHeffteCpuBackend,
  afft_Error_invalidHeffteGpuBackend,
  afft_Error_invalidDftType,
  afft_Error_invalidDhtType,
  afft_Error_invalidDttType,
} afft_Error;

#ifdef __cplusplus
}
#endif

#endif /* AFFT_ERROR_H */
