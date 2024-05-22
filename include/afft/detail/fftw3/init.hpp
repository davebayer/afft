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
#include "../error.hpp"

namespace afft::detail::fftw3
{
  /// @brief Initialize the FFTW3 library.
  inline void init()
  {
    auto check = [](int result)
    {
      if (result == 0)
      {
        throw makeException<std::runtime_error>("[FFTW3 error] initialization failed.");
      }
    };

# ifdef AFFT_FFTW3_HAS_FLOAT
    check(Lib<typePrecision<float>>::initThreads());
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
    check(Lib<typePrecision<double>>::initThreads());
# endif
# ifdef AFFT_FFTW3_HAS_LONG
    check(Lib<typePrecision<long double>>::initThreads());
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
    check(Lib<Precision::f128>::initThreads()); // fixme: precision
# endif

# if AFFT_DISTRIB_IMPL_IS(MPI)
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    MpiLib<typePrecision<float>>::init();
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    MpiLib<typePrecision<double>>::init();
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    MpiLib<typePrecision<long double>>::init();
#   endif
# endif
  }

  /// @brief Finalize the FFTW3 library.
  inline void finalize()
  {
# if AFFT_DISTRIB_IMPL_IS(MPI)
#   ifdef AFFT_FFTW3_HAS_MPI_FLOAT
    MpiLib<typePrecision<float>>::cleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_DOUBLE
    MpiLib<typePrecision<double>>::cleanUp();
#   endif
#   ifdef AFFT_FFTW3_HAS_MPI_LONG
    MpiLib<typePrecision<long double>>::cleanUp();
#   endif
# endif

# ifdef AFFT_FFTW3_HAS_FLOAT
    Lib<typePrecision<float>>::cleanUpThreads();
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
    Lib<typePrecision<double>>::cleanUpThreads();
# endif
# ifdef AFFT_FFTW3_HAS_LONG
    Lib<typePrecision<long double>>::cleanUpThreads();
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
    Lib<Precision::f128>::cleanUpThreads(); // fixme: precision
# endif
  }
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_INIT_HPP */
