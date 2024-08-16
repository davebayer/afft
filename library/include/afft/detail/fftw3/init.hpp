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
#include "../../error.hpp"

namespace afft::detail::fftw3
{
  /// @brief Initialize the FFTW3 library.
  inline void init()
  {
    [[maybe_unused]] auto check = [](int result)
    {
      if (result == 0)
      {
        throw Exception{Error::fftw3, "initialization failed"};
      }
    };

# ifdef AFFT_FFTW3_HAS_FLOAT
#   ifdef AFFT_FFTW3_HAS_FLOAT_THREADS
    check(Lib<afft::fftw3::Library::_float>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    Lib<afft::fftw3::Library::_float>::mpiInit();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
#   ifdef AFFT_FFTW3_HAS_DOUBLE_THREADS
    check(Lib<afft::fftw3::Library::_double>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    Lib<afft::fftw3::Library::_double>::mpiInit();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_LONG
#   ifdef AFFT_FFTW3_HAS_LONG_THREADS
    check(Lib<afft::fftw3::Library::_longDouble>::initThreads());
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    Lib<afft::fftw3::Library::_longDouble>::mpiInit();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
#   ifdef AFFT_FFTW3_HAS_QUAD_THREADS
    check(Lib<afft::fftw3::Library::_quad>::initThreads());
#   endif
# endif
  }

  /// @brief Finalize the FFTW3 library.
  inline void finalize()
  {
# ifdef AFFT_FFTW3_HAS_FLOAT
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    Lib<afft::fftw3::Library::_float>::mpiCleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_FLOAT_THREADS
    Lib<afft::fftw3::Library::_float>::cleanUpThreads();
#   else
    Lib<afft::fftw3::Library::_float>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    Lib<afft::fftw3::Library::_double>::mpiCleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_DOUBLE_THREADS
    Lib<afft::fftw3::Library::_double>::cleanUpThreads();
#   else
    Lib<afft::fftw3::Library::_double>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_LONG
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    Lib<afft::fftw3::Library::longDouble>::mpiCleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_LONG_THREADS
    Lib<afft::fftw3::Library::longDouble>::cleanUpThreads();
#   else
    Lib<afft::fftw3::Library::longDouble>::cleanUp();
#   endif
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
#   ifdef AFFT_FFTW3_HAS_QUAD_THREADS
    Lib<afft::fftw3::Library::quad>::cleanUpThreads();
#   else
    Lib<afft::fftw3::Library::quad>::cleanUp();
#   endif
# endif
  }
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_INIT_HPP */
