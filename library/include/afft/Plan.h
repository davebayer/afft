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
# include <afft/detail/include.h>
#endif

#include <afft/backend.h>
#include <afft/Description.h>
#include <afft/common.h>
#include <afft/error.h>
#include <afft/memory.h>
#include <afft/mp.h>
#include <afft/target.h>
#include <afft/transform.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Plan structure (opaque)
typedef struct afft_Plan afft_Plan;

/**
 * @brief Create a plan.
 * @param desc Plan description.
 * @param backendParams Backend parameters.
 * @param planPtr Pointer to the plan.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_create(const afft_Description* desc,
                            const void*             backendParams,
                            afft_Plan**             planPtr,
                            afft_ErrorDetails*      errDetails);

/**
 * @brief Destroy a plan.
 * @param plan Plan.
 */
void afft_Plan_destroy(afft_Plan* plan);

/**
 * @brief Get the plan description. Do not modify the plan description nor destroy it.
 * @param plan Plan object.
 * @param desc Pointer to the plan description.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_getDescription(const afft_Plan*         plan,
                                    const afft_Description** desc,
                                    afft_ErrorDetails*       errDetails);

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_getBackend(afft_Plan* plan, afft_Backend* backend, afft_ErrorDetails* errDetails);

/**
 * @brief Get the plan workspace size. Only valid if external workspace is used.
 * @param plan Plan object.
 * @param workspaceSizes Pointer to the workspace sizes of target count size.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_getExternalWorkspaceSizes(afft_Plan* plan, const size_t** workspaceSizes, afft_ErrorDetails* errDetails);

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @param execParams Execution parameters.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_Plan_execute(afft_Plan*         plan,
                             void* const*       src,
                             void* const*       dst,
                             const void*        execParams,
                             afft_ErrorDetails* errDetails);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_PLAN_H */
