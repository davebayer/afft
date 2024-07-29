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

#include <afft/afft.hpp>

#include "error.hpp"

/**
 * @brief Make strides.
 * @param shapeRank Rank of the shape.
 * @param shape Shape of the array.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @param errorDetails Error details.
 * @return Error code.
 */
extern "C" afft::c::Error
afft_makeStrides(const size_t           shapeRank,
                 const afft::c::Size*   shape,
                 const afft::c::Size    fastestAxisStride,
                 afft::c::Size*         strides,
                 afft::c::ErrorDetails* errorDetails)
try
{
  if (shapeRank > 0 && shape == nullptr)
  {
    setErrorDetails(errorDetails, "invalid shape");
    return afft_Error_invalidArgument;
  }

  if (fastestAxisStride == 0)
  {
    setErrorDetails(errorDetails, "invalid fastest axis stride");
    return afft_Error_invalidArgument;
  }

  if (shapeRank > 0 && strides == nullptr)
  {
    setErrorDetails(errorDetails, "invalid strides");
    return afft_Error_invalidArgument;
  }

  afft::makeStrides(afft::View<std::size_t>{shape, shapeRank},
                    fastestAxisStride,
                    afft::Span<std::size_t>{strides, shapeRank});

  return afft_Error_success;
}
catch (...)
{
  return handleException(errorDetails);
}

/**
 * @brief Make transposed strides.
 * @param shapeRank Rank of the shape.
 * @param resultShape Shape of the result array.
 * @param orgAxesOrder Original axes order.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param strides Strides of the array.
 * @param errorDetails Error details.
 * @return Error code.
 */
extern "C" afft::c::Error
afft_makeTransposedStrides(const size_t         shapeRank,
                           const afft::c::Size* resultShape,
                           const afft::c::Size* orgAxesOrder,
                           const afft::c::Size  fastestAxisStride,
                           afft::c::Size*       strides,
                           afft::c::ErrorDetails* errorDetails)
try
{
  if (shapeRank > 0 && resultShape == nullptr)
  {
    setErrorDetails(errorDetails, "invalid shape");
    return afft_Error_invalidArgument;
  }

  if (shapeRank > 0 && orgAxesOrder == nullptr)
  {
    setErrorDetails(errorDetails, "invalid original axes order");
    return afft_Error_invalidArgument;
  }

  if (fastestAxisStride == 0)
  {
    setErrorDetails(errorDetails, "invalid fastest axis stride");
    return afft_Error_invalidArgument;
  }

  if (shapeRank > 0 && strides == nullptr)
  {
    setErrorDetails(errorDetails, "invalid strides");
    return afft_Error_invalidArgument;
  }

  afft::makeTransposedStrides(afft::View<std::size_t>{resultShape, shapeRank},
                              afft::View<std::size_t>{orgAxesOrder, shapeRank},
                              fastestAxisStride,
                              afft::Span<std::size_t>{strides, shapeRank});

  return afft_Error_success;
}
catch (...)
{
  return handleException(errorDetails);
}
