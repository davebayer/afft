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

#ifdef MATLABW_ENABLE_GPU
# include <cuda_runtime.h>
#endif

#include "transform.hpp"

using namespace matlabw;

#ifdef MATLABW_ENABLE_GPU
/**
 * @brief Get the current GPU device.
 * @param[in] errorId Error identifier to throw.
 * @return Current GPU device.
 */
static inline int getCurrentGpuDevice(const char* errorId)
{
  int device{};

  if (cudaGetDevice(&device) != cudaSuccess)
  {
    throw mx::Exception{errorId, "failed to get current CUDA device"};
  }

  return device;
}
#endif

/**
 * @brief Check if the shape rank is within the maximum dimension count.
 * @param[in] shapeRank Shape rank to check.
 * @param[in] errorId Error identifier to throw.
 */
static constexpr void checkShapeRank(const std::size_t shapeRank, const char* errorId)
{
  if (shapeRank >= afft::maxDimCount)
  {
    throw mx::Exception{errorId, "input array rank exceeds maximum dimension count"};
  }
}

/**
 * @brief Perform a 1D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void fft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:fft:unimplemented", "not yet implemented");
}

/**
 * @brief Perform a 2D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 */
void fft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:fft2:unimplemented", "not yet implemented");
}

/**
 * @brief Perform an N-dimensional forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 */
void fftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:fftn:unimplemented", "not yet implemented");
}

/**
 * @brief Perform a 1D inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:ifft:unimplemented", "not yet implemented");
}

/**
 * @brief Perform a 2D inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:ifft2:unimplemented", "not yet implemented");
}

/**
 * @brief Perform an N-dimensional inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception("afft:ifftn:unimplemented", "not yet implemented");
}
