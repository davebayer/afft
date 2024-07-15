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

#ifndef AFFT_UTILS_H
#define AFFT_UTILS_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"
#include "error.h"
#include "memory.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Get the alignment of the pointers.
 * @param count Number of pointers.
 * @param ... Pointers.
 * @return Alignment.
 */
static inline afft_Alignment afft_alignmentOf(size_t count, ...)
{
  va_list args;
  va_start(args, count);

  uintptr_t bitOredPtrs = 0;

  for (size_t i = 0; i < count; ++i)
  {
    bitOredPtrs |= va_arg(args, uintptr_t);
  }

  const afft_Alignment alignment = (afft_Alignment)(bitOredPtrs & ~(bitOredPtrs - 1));

  va_end(args);

  return alignment;
}

/**
 * @brief Make strides.
 * @param shapeRank Rank of the shape.
 * @param shape Shape of the array.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @return Error code.
 */
afft_Error afft_makeStrides(const size_t     shapeRank,
                            const afft_Size* shape,
                            const afft_Size  fastestAxisStride,
                            afft_Size*       strides);

/**
 * @brief Make transposed strides.
 * @param shapeRank Rank of the shape.
 * @param resultShape Shape of the result array.
 * @param orgAxesOrder Original axes order.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @return Error code.
 */
afft_Error afft_makeTransposedStrides(const size_t     shapeRank,
                                      const afft_Size* resultShape,
                                      const afft_Size* orgAxesOrder,
                                      const afft_Size  fastestAxisStride,
                                      afft_Size*       strides);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_UTILS_H */
