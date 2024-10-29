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

#ifndef AFFT_PLAN_CACHE_H
#define AFFT_PLAN_CACHE_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.h>
#endif

#include <afft/common.h>
#include <afft/error.h>
#include <afft/Plan.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Opaque plan cache structure
typedef struct _afft_PlanCache afft_PlanCache;

/**
 * @brief Creates a new plan cache.
 * @param planCache The new plan cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_create(afft_PlanCache** planCache, afft_ErrorDetails* errorDetails);

/**
 * @brief Destroys a plan cache.
 * @param planCache The plan cache.
 */
void afft_PlanCache_destroy(afft_PlanCache* planCache);

/**
 * @brief Is the cache empty?
 * @param planCache The plan cache.
 * @param isEmpty Is the cache empty?
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_isEmpty(afft_PlanCache* planCache, bool* isEmpty, afft_ErrorDetails* errorDetails);

/**
 * @brief Gets the current size of the cache.
 * @param planCache The plan cache.
 * @param size The current size of the cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_getSize(afft_PlanCache* planCache, size_t* size, afft_ErrorDetails* errorDetails);

/**
 * @brief Gets the current size of the cache.
 * @param planCache The plan cache.
 * @param size The current size of the cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_getMaxSize(afft_PlanCache* planCache, size_t* maxSize, afft_ErrorDetails* errorDetails);

/**
 * @brief Sets the maximum size of the cache.
 * @param planCache The plan cache.
 * @param maxSize The maximum size of the cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_setMaxSize(afft_PlanCache* planCache, size_t maxSize, afft_ErrorDetails* errorDetails);

/**
 * @brief Clears the cache.
 * @param planCache The plan cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_clear(afft_PlanCache* planCache, afft_ErrorDetails* errorDetails);

/**
 * @brief Inserts a plan into the cache. If the cache is full, the least recently used plan is removed.
 *        The plan ownership is transferred to the cache.
 * @param planCache The plan cache.
 * @param plan The plan to insert.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_insert(afft_PlanCache* planCache, afft_Plan* plan, afft_ErrorDetails* errorDetails);

// TODO
afft_Error afft_PlanCache_erase(...);

// TODO
afft_Error afft_PlanCache_release(...);

/**
 * @brief Merges two plan caches. The plans are moved from the other plan cache to the plan cache.
 * @param planCache The plan cache.
 * @param otherPlanCache The other plan cache.
 * @param errorDetails Error details.
 * @return Error code.
 */
afft_Error afft_PlanCache_merge(afft_PlanCache* planCache, afft_PlanCache* otherPlanCache, afft_ErrorDetails* errorDetails);

// TODO
afft_Error afft_PlanCache_find(...);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_PLAN_CACHE_H */
