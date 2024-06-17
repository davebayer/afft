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

#include <afft/afft.h>
#include <afft/afft.hpp>

/**
 * @brief Make strides.
 * @param shapeRank Rank of the shape.
 * @param shape Shape of the array.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @return Error code.
 */
extern "C" afft_Error afft_makeStrides(const size_t  shapeRank,
                                       const size_t* shape,
                                       const size_t  fastestAxisStride,
                                       size_t*       strides)
try
{
  if (shapeRank > 0 && shape == nullptr)
  {
    return afft_Error_invalidShape;
  }

  if (shapeRank > 0 && strides == nullptr)
  {
    return afft_Error_invalidStrides;
  }

  afft::makeStrides(afft::View<std::size_t>{shape, shapeRank},
                    fastestAxisStride,
                    afft::Span<std::size_t>{strides, shapeRank});

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Make transposed strides.
 * @param shapeRank Rank of the shape.
 * @param resultShape Shape of the result array.
 * @param orgAxesOrder Original axes order.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @return Error code.
 */
extern "C" afft_Error afft_makeTransposedStrides(const size_t  shapeRank,
                                                 const size_t* resultShape,
                                                 const size_t* orgAxesOrder,
                                                 const size_t  fastestAxisStride,
                                                 size_t*       strides)
try
{
  if (shapeRank > 0 && resultShape == nullptr)
  {
    return afft_Error_invalidShape;
  }

  if (shapeRank > 0 && orgAxesOrder == nullptr)
  {
    return afft_Error_invalidAxes;
  }

  if (shapeRank > 0 && strides == nullptr)
  {
    return afft_Error_invalidStrides;
  }

  afft::makeTransposedStrides(afft::View<std::size_t>{resultShape, shapeRank},
                              afft::View<std::size_t>{orgAxesOrder, shapeRank},
                              fastestAxisStride,
                              afft::Span<std::size_t>{strides, shapeRank});

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}
