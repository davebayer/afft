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

#ifndef AFFT_DETAIL_FFTW3_INIT_HPP
#define AFFT_DETAIL_FFTW3_INIT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "Lib.hpp"
#include "../../exception.hpp"

namespace afft::detail::fftw3
{
  /// @brief Initialize the FFTW3 library.
  inline void init()
  {
    [[maybe_unused]] auto check = [](int result)
    {
      if (result == 0)
      {
        throw BackendError{Backend::fftw3, "initialization failed."};
      }
    };

# ifdef AFFT_FFTW3_HAS_FLOAT
#   ifdef AFFT_FFTW3_HAS_FLOAT_THREADS
    check(Lib<Precision::_float>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    MpiLib<Precision::_float>::init();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
#   ifdef AFFT_FFTW3_HAS_DOUBLE_THREADS
    check(Lib<Precision::_double>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    MpiLib<Precision::_double>::init();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_LONG
#   ifdef AFFT_FFTW3_HAS_LONG_THREADS
    check(Lib<Precision::_longDouble>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    MpiLib<Precision::_longDouble>::init();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
#   ifdef AFFT_FFTW3_HAS_QUAD_THREADS
    check(Lib<Precision::_quad>::initThreads());
#   endif
# endif
  }

  /// @brief Finalize the FFTW3 library.
  inline void finalize()
  {
# ifdef AFFT_FFTW3_HAS_FLOAT
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    MpiLib<Precision::_float>::cleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_FLOAT_THREADS
    Lib<Precision::_float>::cleanUpThreads();
#   else
    Lib<Precision::_float>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    MpiLib<Precision::_double>::cleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_DOUBLE_THREADS
    Lib<Precision::_double>::cleanUpThreads();
#   else
    Lib<Precision::_double>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_LONG
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    MpiLib<Precision::_longDouble>::cleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_LONG_THREADS
    Lib<Precision::_longDouble>::cleanUpThreads();
#   else
    Lib<Precision::_longDouble>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
#   ifdef AFFT_FFTW3_HAS_QUAD_THREADS
    Lib<Precision::_quad>::cleanUpThreads();
#   else
    Lib<Precision::_quad>::cleanUp();
#   endif
# endif
  }
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_INIT_HPP */
