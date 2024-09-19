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

#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <afft/afft.hpp>
#include <matlabw/mx/mx.hpp>

/**
 * @brief Perform a 1D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void fft(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 */
void fft2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 */
void fftn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

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
void ifft(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

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
void ifft2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifftn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 1D forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void rfft(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 */
void rfft2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 */
void rfftn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 1D inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void irfft(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 */
void irfft2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 */
void irfftn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 1D discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void dct(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void dct2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dct type.
 */
void dctn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 1D inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void idct(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void idct2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dct type.
 */
void idctn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 1D discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dst type.
 */
void dst(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform a 2D discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 *            _ 'type' + value specifies the dst type.
 */
void dst2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

/**
 * @brief Perform an N-dimensional discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dst type.
 */
void dstn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs);

#endif /* TRANSFORM_HPP */
