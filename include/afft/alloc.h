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

#ifndef AFFT_ALLOC_H
#define AFFT_ALLOC_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Allocate aligned memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @param alignment Alignment of the memory block.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
void* afft_cpu_alignedAlloc(size_t sizeInBytes, afft_Alignment alignment);

/**
 * @brief Free aligned memory.
 * @param ptr Pointer to the memory block.
 * @param alignment Alignment of the memory block.
 */
void afft_cpu_alignedFree(void* ptr, afft_Alignment alignment);

#if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
/**
 * @brief Allocate unified memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
void* afft_gpu_unifiedAlloc(size_t sizeInBytes);

/**
 * @brief Free unified memory.
 * @param ptr Pointer to the memory block.
 */
void afft_gpu_unifiedFree(void* ptr);
#elif AFFT_GPU_BACKEND_IS(OPENCL)
/**
 * @brief Allocate unified memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @param context OpenCL context.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
void* afft_gpu_unifiedAlloc(size_t sizeInBytes, cl_context context);

/**
 * @brief Free unified memory.
 * @param ptr Pointer to the memory block.
 * @param context OpenCL context.
 */
void afft_gpu_unifiedFree(void* ptr, cl_context context);
#endif

#ifdef __cplusplus
}
#endif

#endif /* AFFT_ALLOC_H */
