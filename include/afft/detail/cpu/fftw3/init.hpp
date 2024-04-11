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

#ifndef AFFT_DETAIL_CPU_FFTW3_INIT_HPP
#define AFFT_DETAIL_CPU_FFTW3_INIT_HPP

#include <fftw3.h>

namespace afft::detail::cpu::fftw3
{
  /// @brief Initialize the FFTW3 library.
  inline void init()
  {
    fftwf_init_threads();
    fftw_init_threads();
    fftwl_init_threads();
  }

  /// @brief Finalize the FFTW3 library.
  inline void finalize()
  {
    fftwf_cleanup_threads();
    fftw_cleanup_threads();
    fftwl_cleanup_threads();
  }
} // namespace afft::detail::cpu::fftw3

#endif /* AFFT_DETAIL_CPU_FFTW3_INIT_HPP */
