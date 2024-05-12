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

#ifndef AFFT_MP_HPP
#define AFFT_MP_HPP

#include "macro.hpp"

/// @brief Disable multi-process backend
#define AFFT_MP_BACKEND_NONE (0)
/// @brief MPI multi-process backend
#define AFFT_MP_BACKEND_MPI  (1)

/// @brief Check if multi-process is enabled
#define AFFT_MP_IS_ENABLED   (AFFT_MP_BACKEND_IS(NONE))

/**
 * @brief Check if multi-process backend is enabled
 * @param bckndName multi-process backend name
 * @return true if multi-process backend is enabled, false otherwise
 */
#define AFFT_MP_BACKEND_IS(bckndName) \
  (AFFT_DETAIL_EXPAND_AND_CONCAT(AFFT_MP_BACKEND_, AFFT_MP_BACKEND) == AFFT_MP_BACKEND_##bckndName)

// Check if multi-process backend is supported
#ifdef AFFT_MP_BACKEND
# if !(AFFT_MP_BACKEND_IS(MPI))
#  error "Unsupported multi-process backend"
# endif
#else
# define AFFT_MP_BACKEND     NONE
#endif

// Include distribution type headers
#if AFFT_MP_BACKEND_IS(MPI)
# include <mpi.h>
#endif

namespace afft
{
  /**
   * @struct MultiProcessParameters
   * @brief Multi-process parameters
   */
  struct MultiProcessParameters
#if AFFT_MP_IS_ENABLED
  {
# if AFFT_MP_BACKEND_IS(MPI)
    MPI_Comm communicator{MPI_COMM_WORLD}; ///< MPI communicator
# endif
  }
#endif
   ;
} // namespace afft

#endif /* AFFT_MP_HPP */
