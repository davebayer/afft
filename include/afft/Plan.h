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
#ifndef AFFT_PLAN_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "architecture.h"
#include "backend.h"
#include "error.h"
#include "common.h"
#include "transform.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Opaque plan structure
typedef struct _afft_Plan afft_Plan;

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  /**
   * @brief Create a plan for a given transform and architecture.
   * @param transformParams Transform parameters. May be any of afft_*transform*_Parameters or general parameters.
   * @param archParams Architecture parameters. May be any of afft_*target*_*distribution*_Parameters or general parameters.
   * @param planPtr Pointer to the plan.
   * @return Error code.
   */
# define afft_Plan_create(transformParams, archParams, planPtr) \
    _afft_Plan_create(afft_makeTransformParameters(transformParams), \
                      afft_makeArchitectureParameters(archParams), \
                      planPtr)
#else
  /**
   * @brief Create a plan for a given transform and architecture.
   * @param transformParams Transform parameters.
   * @param archParams Architecture parameters.
   * @param planPtr Pointer to the plan.
   * @return Error code.
   */
# define afft_Plan_create(transformParams, archParams, planPtr) \
    _afft_Plan_create(transformParams, archParams, planPtr)
#endif

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  /**
   * @brief Create a plan for a given transform, architecture, and backend.
   * @param transformParams Transform parameters. May be any of afft_*transform*_Parameters or general parameters.
   * @param archParams Architecture parameters. May be any of afft_*target*_*distribution*_Parameters or general parameters.
   * @param backendParams Backend parameters. May be any of afft_*target*_*distribution*_Parameters or general parameters.
   * @param planPtr Pointer to the plan.
   * @return Error code.
   */
# define afft_Plan_create(transformParams, archParams, backendParams, planPtr) \
    _afft_Plan_createWithBackendParameters(afft_makeTransformParameters(transformParams), \
                                           afft_makeArchitectureParameters(archParams), \
                                           afft_makeBackendParameters(backendParams), \
                                           planPtr)
#else
  /**
   * @brief Create a plan for a given transform, architecture, and backend.
   * @param transformParams Transform parameters.
   * @param archParams Architecture parameters.
   * @param backendParams Backend parameters.
   * @param planPtr Pointer to the plan.
   * @return Error code.
   */
# define afft_Plan_create(transformParams, archParams, backendParams, planPtr) \
    _afft_Plan_createWithBackendParameters(transformParams, archParams, backendParams, planPtr)
#endif

/**
 * @brief Destroy a plan.
 * @param plan Plan.
 */
void afft_Plan_destroy(afft_Plan* plan);

/**
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTransform(const afft_Plan* plan, afft_Transform* transform);

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTarget(const afft_Plan* plan, afft_Target* target);

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @return Error code.
 */
afft_Error afft_Plan_getTargetCount(const afft_Plan* plan, size_t* targetCount);

/**
 * @brief Get the plan distribution.
 * @param plan Plan object.
 * @param distribution Pointer to the distribution variable.
 * @return Error code.
 */
afft_Error afft_Plan_getDistribution(const afft_Plan* plan, afft_Distribution* distribution);

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @return Error code.
 */
afft_Error afft_Plan_getBackend(const afft_Plan* plan, afft_Backend* backend);

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param workspaceSize Pointer to the workspace array the same size as number of targets.
 * @return Error code.
 */
afft_Error afft_Plan_getWorkspaceSize(const afft_Plan* plan, size_t* workspaceSize);

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @return Error code.
 */
afft_Error afft_Plan_execute(afft_Plan* plan, void* const* src, void* const* dst);

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  /**
   * @brief Execute a plan with execution parameters.
   * @param plan Plan object.
   * @param src Source data pointer array of target count size (x2 if planar complex).
   * @param dst Destination data pointer array of target count size (x2 if planar complex).
   * @param execParams Execution parameters. Any of afft_*target*_*distribution*_Parameters or generic parameters.
   * @return Error code.
   */
# define afft_Plan_executeWithParameters(plan, src, dst, execParams) \
    _afft_Plan_executeWithParameters(plan, src, dst, afft_makeExecutionParameters(execParams))
#else
# define afft_Plan_executeWithParameters(plan, src, dst, execParams) \
    _afft_Plan_executeWithParameters(plan, src, dst, execParams)
#endif

/**********************************************************************************************************************/
// Private functions
/**********************************************************************************************************************/
afft_Error _afft_Plan_create(afft_TransformParameters    transformParams,
                             afft_ArchitectureParameters archParams,
                             afft_Plan**                 planPtr);

afft_Error _afft_Plan_createWithBackendParameters(afft_TransformParameters    transformParams,
                                                  afft_ArchitectureParameters archParams,
                                                  afft_BackendParameters      backendParams,
                                                  afft_Plan**                 planPtr);

afft_Error _afft_Plan_executeWithParameters(afft_Plan* plan, void* const* src, void* const* dst);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_PLAN_H */
