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

#ifndef AFFT_DETAIL_ROCFFT_INIT_HPP
#define AFFT_DETAIL_ROCFFT_INIT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

namespace afft::detail::rocfft
{
  /// @brief Initialize the rocFFT library.
  void init();

  /// @brief Finalize the rocFFT library.
  void finalize();
} // namespace afft::detail::rocfft

#ifdef AFFT_HEADER_ONLY

#include "error.hpp"

namespace afft::detail::rocfft
{
  /// @brief Initialize the rocFFT library.
  AFFT_HEADER_ONLY_INLINE void init()
  {
    checkError(rocfft_setup());
  }

  /// @brief Finalize the rocFFT library.
  AFFT_HEADER_ONLY_INLINE void finalize()
  {
    checkError(rocfft_cleanup());
  }
} // namespace afft::detail::rocfft

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_ROCFFT_INIT_HPP */
