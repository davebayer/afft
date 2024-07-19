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

#ifndef AFFT_PLAN_H
#define AFFT_PLAN_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "backend.h"
#include "error.h"
#include "common.h"
#include "memory.h"
#include "mp.h"
#include "target.h"
#include "transform.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Opaque plan structure
typedef struct _afft_Plan afft_Plan;

/// @brief Plan parameters structure
typedef struct afft_Plan_Parameters afft_Plan_Parameters;

/// @brief Plan parameters structure
struct afft_Plan_Parameters
{
  afft_Transform transform;       ///< Transform type
  afft_MpBackend mpBackend;       ///< Multi-process backend
  afft_Target    target;          ///< Target architecture
  void*          transformParams; ///< Transform parameters
  void*          mpBackendParams; ///< Multi-process backend parameters
  void*          targetParams;    ///< Target parameters
  void*          memoryLayout;    ///< Memory layout
  void*          backendParams;   ///< Backend parameters
};

/**
 * @brief Create a plan for given parameters.
 * @param planParams Plan parameters.
 * @param planPtr Pointer to the plan.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_create(afft_Plan_Parameters planParams, afft_Plan** planPtr, afft_ErrorDetails* errorDetails);

/**
 * @brief Destroy a plan.
 * @param plan Plan.
 */
void afft_Plan_destroy(afft_Plan* plan);

/**
 * @brief Get the plan multi-process backend.
 * @param plan Plan object.
 * @param mpBackend Pointer to the multi-process backend variable.
 * @return Error code.
 */
afft_Error afft_Plan_getMpBackend(afft_Plan* plan, afft_MpBackend* mpBackend);

/**
 * @brief Get the plan multi-process backend parameters.
 * @param plan Plan object.
 * @param mpBackendParams Pointer to the multi-process backend parameters.
 * @return Error code.
 */
afft_Error afft_Plan_getMpBackendParameters(afft_Plan* plan, void* mpBackendParams);

/**
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTransform(afft_Plan* plan, afft_Transform* transform);

/**
 * @brief Get the plan transform parameters.
 * @param plan Plan object.
 * @param transformParams Pointer to the transform parameters.
 * @return Error code.
 */
afft_Error afft_Plan_getTransformParameters(afft_Plan* plan, void* transformParams);

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTarget(afft_Plan* plan, afft_Target* target);

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTargetCount(afft_Plan* plan, size_t* targetCount);

/**
 * @brief Get the plan target parameters.
 * @param plan Plan object.
 * @param targetParams Pointer to the target parameters.
 * @return Error code.
 */
afft_Error afft_Plan_getTargetParameters(afft_Plan* plan, void* targetParams);

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @return Error code.
 */
afft_Error afft_Plan_getBackend(afft_Plan* plan, afft_Backend* backend);

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param workspaceSizes Pointer to the workspace sizes of target count size.
 * @return Error code.
 */
afft_Error afft_Plan_getWorkspaceSizes(afft_Plan* plan, const size_t** workspaceSizes);

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @param execParams Execution parameters.
 * @return Error code.
 */
afft_Error afft_Plan_execute(afft_Plan* plan, void* const* src, void* const* dst, const void* execParams);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_PLAN_H */
