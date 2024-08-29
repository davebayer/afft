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

#ifndef AFFT_DESCRIPTION_H
#define AFFT_DESCRIPTION_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "backend.h"
#include "common.h"
#include "error.h"
#include "memory.h"
#include "mp.h"
#include "target.h"
#include "transform.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Opaque plan description type.
typedef struct afft_Description afft_Description;

/**
 * @brief Create plan description.
 * @param[in]  transform       Transform
 * @param[in]  mpBackend       Multi-process backend
 * @param[in]  target          Target
 * @param[in]  transformParams Transform parameters
 * @param[in]  mpBackendParams Multi-process backend parameters
 * @param[in]  targetParams    Target parameters
 * @param[in]  memoryLayout    Memory layout
 * @param[out] desc            Plan description
 * @param[out] errorDetails    Error details
 * @return Error code
 */
afft_Error afft_Description_create(afft_Transform     transform,
                                   afft_MpBackend     mpBackend,
                                   afft_Target        target,
                                   const void*        transformParams,
                                   const void*        mpBackendParams,
                                   const void*        targetParams,
                                   const void*        memoryLayout,
                                   afft_Description** desc,
                                   afft_ErrorDetails* errorDetails);

/**
 * @brief Destroy plan description.
 * @param[in] desc Plan description
 */
void afft_Description_destroy(afft_Description* desc);

/**
 * @brief Get transform.
 * @param[in]  desc         Plan description
 * @param[out] transform    Transform
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getTransform(const afft_Description* desc,
                                         afft_Transform*         transform,
                                         afft_ErrorDetails*      errorDetails);

/**
 * @brief Get transform parameters.
 * @param[in]  desc            Plan description
 * @param[out] transformParams Transform parameters
 * @param[out] errorDetails    Error details
 * @return Error code
 */
afft_Error afft_Description_getTransformParameters(const afft_Description* desc,
                                                   void*                   transformParams,
                                                   afft_ErrorDetails*      errorDetails);

/**
 * @brief Get transform parameters variant.
 * @param[in]  desc                   Plan description
 * @param[out] transformParamsVariant Transform parameters variant
 * @param[out] errorDetails           Error details
 * @return Error code
 */
afft_Error afft_Description_getTransformParametersVariant(const afft_Description*          desc,
                                                          afft_TransformParametersVariant* transformParamsVariant,
                                                          afft_ErrorDetails*               errorDetails);

/**
 * @brief Get multi-process backend.
 * @param[in]  desc         Plan description
 * @param[out] mpBackend    Multi-process backend
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getMpBackend(const afft_Description* desc,
                                         afft_MpBackend*         mpBackend,
                                         afft_ErrorDetails*      errorDetails);

/**
 * @brief Get multi-process backend parameters.
 * @param[in]  desc            Plan description
 * @param[out] mpBackendParams Multi-process backend parameters
 * @param[out] errorDetails    Error details
 * @return Error code
 */
afft_Error afft_Description_getMpBackendParameters(const afft_Description* desc,
                                                   void*                   mpBackendParams,
                                                   afft_ErrorDetails*      errorDetails);

/**
 * @brief Get multi-process backend parameters variant.
 * @param[in]  desc                   Plan description
 * @param[out] mpBackendParamsVariant Multi-process backend parameters variant
 * @param[out] errorDetails           Error details
 * @return Error code
 */
afft_Error afft_Description_getMpBackendParametersVariant(const afft_Description*          desc,
                                                          afft_MpBackendParametersVariant* mpBackendParamsVariant,
                                                          afft_ErrorDetails*               errorDetails);

/**
 * @brief Get target.
 * @param[in]  desc         Plan description
 * @param[out] target       Target
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getTarget(const afft_Description* desc,
                                      afft_Target*            target,
                                      afft_ErrorDetails*      errorDetails);

/**
 * @brief Get target count.
 * @param[in]  desc         Plan description
 * @param[out] targetCount  Target count
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getTargetCount(const afft_Description* desc,
                                           size_t*                 targetCount,
                                           afft_ErrorDetails*      errorDetails);

/**
 * @brief Get target parameters.
 * @param[in]  desc         Plan description
 * @param[out] targetParams Target parameters
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getTargetParameters(const afft_Description* desc,
                                                void*                   targetParams,
                                                afft_ErrorDetails*      errorDetails);

/**
 * @brief Get target parameters variant.
 * @param[in]  desc                Plan description
 * @param[out] targetParamsVariant Target parameters variant
 * @param[out] errorDetails        Error details
 * @return Error code
 */
afft_Error afft_Description_getTargetParametersVariant(const afft_Description*       desc,
                                                       afft_TargetParametersVariant* targetParamsVariant,
                                                       afft_ErrorDetails*            errorDetails);

/**
 * @brief Get memory layout.
 * @param[in]  desc         Plan description
 * @param[out] memoryLayout Memory layout
 * @param[out] errorDetails Error details
 * @return Error code
 */
afft_Error afft_Description_getMemoryLayout(const afft_Description* desc,
                                            afft_MemoryLayout*      memoryLayout,
                                            afft_ErrorDetails*      errorDetails);

/**
 * @brief Get memory layout parameters.
 * @param[in]  desc               Plan description
 * @param[out] memoryLayoutParams Memory layout parameters
 * @param[out] errorDetails       Error details
 * @return Error code
 */
afft_Error afft_Description_getMemoryLayoutParameters(const afft_Description* desc,
                                                      void*                   memoryLayoutParams,
                                                      afft_ErrorDetails*      errorDetails);

/**
 * @brief Get memory layout parameters variant.
 * @param[in]  desc                      Plan description
 * @param[out] memoryLayoutParamsVariant Memory layout parameters variant
 * @param[out] errorDetails              Error details
 * @return Error code
 */
afft_Error afft_Description_getMemoryLayoutParametersVariant(const afft_Description*             desc,
                                                             afft_MemoryLayoutParametersVariant* memoryLayoutParamsVariant,
                                                             afft_ErrorDetails*                  errorDetails);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_DESCRIPTION_H */
