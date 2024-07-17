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

#ifndef HELPERS_MPI_HPP
#define HELPERS_MPI_HPP

#include <stdexcept>

#include <mpi.h>

#include "cformat.hpp"

/**
 * @brief Macro for checking MPI errors. The call cannot contain _error variable.
 * @param call MPI function call
 */
#define MPI_CALL(call) \
  do { \
    const int _error = (call); \
    if (_error != MPI_SUCCESS) \
    { \
      throw cformatNothrow("MPI error (%s:%d) - error code #%d", __FILE__, __LINE__, _error); \
    } \
  } while (0)

namespace helpers::mpi
{
  /**
   * @brief Get the rank of the MPI process in the communicator.
   * @param comm MPI communicator
   * @return Rank of the MPI process
   */
  inline int getRank(MPI_Comm comm)
  {
    int rank{};

    MPI_CALL(MPI_Comm_rank(comm, &rank));

    return rank;
  }
} // namespace helpers::mpi

#endif /* HELPERS_MPI_HPP */
