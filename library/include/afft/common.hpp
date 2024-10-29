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

#ifndef AFFT_COMMON_HPP
#define AFFT_COMMON_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.hpp>
#endif

AFFT_EXPORT namespace afft
{
  /// @brief Maximum number of dimensions
  inline constexpr std::size_t maxRank{AFFT_MAX_RANK};

  /// @brief Order type
  enum class Order : std::underlying_type_t<::afft_Order>
  {
    rowMajor    = afft_Order_rowMajor,    ///< Row-major order
    columnMajor = afft_Order_columnMajor, ///< Column-major order
  };

  /// @brief Axis type
  using Axis = ::afft_Axis;

  /// @brief Size type
  using Size = ::afft_Size;

  /// @brief Stride type
  using Stride = ::afft_Stride;

  namespace fftw3
  {
    /// @brief FFTW3 library type
    enum class Library : std::underlying_type_t<::afft_fftw3_Library>;
  };

  /// @brief FFTW3 library type
  enum class fftw3::Library : std::underlying_type_t<::afft_fftw3_Library>
  {
    _float     = afft_fftw3_Library_float,      ///< FFTW3 single precision (fftwf)
    _double    = afft_fftw3_Library_double,     ///< FFTW3 double precision (fftw)
    longDouble = afft_fftw3_Library_longDouble, ///< FFTW3 long double precision (fftwl)
    quad       = afft_fftw3_Library_quad,       ///< FFTW3 quadruple precision (fftwq)
  };
} // namespace afft

#endif /* AFFT_COMMON_HPP */
