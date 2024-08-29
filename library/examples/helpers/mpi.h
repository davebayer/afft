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

#ifndef HELPERS_MPI_H
#define HELPERS_MPI_H

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Check MPI error and exit if not success. Should not be used directly, use MPI_CALL macro instead.
 * @param[in] error MPI error
 * @param[in] file  file name
 * @param[in] line  line number
 */
static inline void check_mpi_error(int error, const char* file, int line)
{
  if (error != MPI_SUCCESS)
  {
    fprintf(stderr, "MPI error (%s:%d) - error code #%d\n", file, line, error);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Macro for checking MPI errors. The call cannot contain _error variable.
 * @param[in] call MPI function call
 */
#define MPI_CALL(call) check_mpi_error((call), __FILE__, __LINE__)

/**
 * @brief Get the rank of the MPI process in the communicator.
 * @param[in] comm MPI communicator
 * @return Rank of the MPI process
 */
static inline int helpers_mpi_getRank(MPI_Comm comm)
{
  int rank{};

  MPI_CALL(MPI_Comm_rank(comm, &rank));

  return rank;
}

#ifdef __cplusplus
}
#endif

#endif /* HELPERS_MPI_H */
