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

#ifndef AFFT_DETAIL_CPU_INIT_HPP
#define AFFT_DETAIL_CPU_INIT_HPP

#include "../../cpu.hpp"

#if AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(FFTW3)
# include "fftw3/init.hpp"
#endif
#if AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(MKL)
# include "mkl/init.hpp"
#endif
#if AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(POCKETFFT)
# include "pocketfft/init.hpp"
#endif

namespace afft::detail::cpu
{
  /// @brief Initialize the CPU transform backend.
  inline void init()
  {
# if AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(FFTW3)
    fftw3::init();
# elif AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(MKL)
    mkl::init();
# elif AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(POCKETFFT)
    pocketfft::init();
# endif
  }

  /// @brief Finalize the CPU transform backend.
  inline void finalize()
  {
# if AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(FFTW3)
    fftw3::finalize();
# elif AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(MKL)
    mkl::finalize();
# elif AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(POCKETFFT)
    pocketfft::finalize();
# endif
  }
} // namespace afft::detail::cpu

#endif /* AFFT_DETAIL_CPU_INIT_HPP */
