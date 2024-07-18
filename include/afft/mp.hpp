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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#ifdef AFFT_ENABLE_MPI
# include "detail/mpi/mpi.hpp"
#endif

AFFT_EXPORT namespace afft
{
  /// @brief Multi-process backend
  enum class MpBackend : std::uint8_t
  {
    none, ///< No multi-process backend (single-process)
    mpi,  ///< MPI multi-process backend
  };

  /**
   * @brief Multi-process backend constant
   * @tparam _mpBackend Multi-process backend
   */
  template<MpBackend _mpBackend>
  struct MpBackendConstant
  {
    static constexpr MpBackend mpBackend = _mpBackend;
  };

  /// @brief Single-process parameters
  struct SingleProcessParameters;

  namespace mpi
  {
    /// @brief MPI multi-process parameters
    struct Parameters;
  } // namespace mpi

  /// @brief Single-process parameters
  struct SingleProcessParameters : MpBackendConstant<MpBackend::none> {};

#ifdef AFFT_ENABLE_MPI
  /// @brief MPI multi-process parameters
  struct mpi::Parameters : MpBackendConstant<MpBackend::mpi>
  {
    MPI_Comm comm{MPI_COMM_WORLD}; ///< MPI communicator
  };
#endif
} // namespace afft

#endif /* AFFT_MP_HPP */
