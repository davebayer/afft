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

#ifndef AFFT_MP_H
#define AFFT_MP_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Multi-process backend
typedef uint8_t afft_MpBackend;

/**
 * @brief Multi-process backend
 */
#define afft_MpBackend_none (afft_MpBackend)0 ///< No multi-process backend (single-process)
#define afft_MpBackend_mpi  (afft_MpBackend)1 ///< MPI multi-process backend

/// @brief MPI multi-process parameters
typedef struct afft_mpi_Parameters afft_mpi_Parameters;

#ifdef AFFT_ENABLE_MPI
/// @brief MPI multi-process parameters
struct afft_mpi_Parameters
{
  MPI_Comm comm; ///< MPI communicator
};
#endif

#ifdef __cplusplus
}
#endif

#endif /* AFFT_MP_H */
