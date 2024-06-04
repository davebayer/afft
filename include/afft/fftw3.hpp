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

#ifndef AFFT_FFTW3_HPP
#define AFFT_FFTW3_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "exception.hpp"
#include "typeTraits.hpp"
#if AFFT_BACKEND_IS_ENABLED(FFTW3)
# include "detail/fftw3/Lib.hpp"
#endif

AFFT_EXPORT namespace afft::fftw3
{
  /**
   * @brief Does the FFTW3 library support the given precision?
   * @tparam prec Precision of the FFTW3 library.
   */
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
  template<Precision prec>
  inline constexpr bool isSupportedPrecision = detail::fftw3::IsSupportedPrecision<prec>::value;
# else
  template<Precision prec>
  inline constexpr bool isSupportedPrecision = false;
# endif

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param filename Name of the file to export the wisdom to.
   */
  template<typename PrecT>
  void exportWisdomToFilename([[maybe_unused]] std::string_view filename)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::Lib<typePrecision<PrecT>>::exportWisdomToFilename(filename.data()))
      {
        throw BackendError{Backend::fftw3, "failed to export wisdom to file"};
      }
    }
# endif
  }

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param file File to export the wisdom to.
   */
  template<typename PrecT>
  void exportWisdomToFile([[maybe_unused]] FILE* file)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::Lib<typePrecision<PrecT>>::exportWisdomToFile(file))
      {
        throw BackendError{Backend::fftw3, "failed to export wisdom to file"};
      }
    }
# endif
  }

  /**
   * @brief Export FFTW3 wisdom to a string.
   * @tparam PrecT Precision of the FFTW3 library.
   * @return String containing the wisdom.
   */
  template<typename PrecT>
  [[nodiscard]] std::string exportWisdom()
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

    std::string wisdom{};

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      struct FreeDeleter
      {
        void operator()(char* ptr) const
        {
          free(ptr);
        }
      };

      std::unique_ptr<char, FreeDeleter> orgWisdom{detail::fftw3::Lib<typePrecision<PrecT>>::exportWisdomToString()};

      if (orgWisdom)
      {
        wisdom = orgWisdom.get();
      }
    }
# endif

    return wisdom;
  }

  /**
   * @brief Import FFTW3 wisdom from the system. Only on Unix and GNU systems.
   * @tparam PrecT Precision of the FFTW3 library.
   */
  template<typename PrecT>
  void importSystemWisdom()
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::MpiLib<typePrecision<PrecT>>::importSystemWisdom())
      {
        throw BackendError{Backend::fftw3, "failed to import system wisdom"};
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param filename Name of the file to import the wisdom from.
   */
  template<typename PrecT>
  void importWisdomFromFilename([[maybe_unused]] std::string_view filename)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::MpiLib<typePrecision<PrecT>>::importWisdomFromFilename(filename.data()))
      {
        throw BackendError{Backend::fftw3, "failed to import wisdom from file"};
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param file File to import the wisdom from.
   */
  template<typename PrecT>
  void importWisdomFromFile([[maybe_unused]] FILE* file)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::Lib<typePrecision<PrecT>>::importWisdomFromFile(file))
      {
        throw BackendError{Backend::fftw3, "failed to import wisdom from file"};
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param wisdom String containing the wisdom.
   */
  template<typename PrecT>
  void importWisdom([[maybe_unused]] std::string_view wisdom)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      if (!detail::fftw3::Lib<typePrecision<PrecT>>::importWisdomFromString(wisdom.data()))
      {
        throw BackendError{Backend::fftw3, "failed to import wisdom"};
      }
    }
# endif
  }

  /**
   * @brief Forget all FFTW3 wisdom.
   * @tparam PrecT Precision of the FFTW3 library.
   */
  template<typename PrecT>
  void forgetWisdom()
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasPrecision<typePrecision<PrecT>>)
    {
      detail::fftw3::Lib<typePrecision<PrecT>>::forgetWisdom();
    }
# endif
  }

namespace mpst
{
  /**
   * @brief Does the FFTW3 MPI library support the given precision?
   * @tparam prec Precision of the FFTW3 MPI library.
   */
#if AFFT_BACKEND_IS_ENABLED(FFTW3) && AFFT_MP_BACKEND_IS(MPI)
  template<Precision prec>
  inline constexpr bool isSupportedPrecision = detail::fftw3::IsMpiSupportedPrecision<prec>::value;
#else
  template<Precision prec>
  inline constexpr bool isSupportedPrecision = false;
#endif

#if AFFT_MP_BACKEND_IS(MPI)
  /**
   * @brief Broadcast FFTW3 wisdom to all MPI processes from the root process.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param comm MPI communicator.
   */
  template<typename PrecT>
  void broadcastWisdom([[maybe_unused]] MPI_Comm comm)
  {
    static_assert(isSupportedPrecision<typePrecision<PrecT>>, "Unsupported FFTW3 precision");

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasMpiPrecision<typePrecision<PrecT>>)
    {
      detail::fftw3::MpiLib<typePrecision<PrecT>>::broadcastWisdom(comm);
    }
# endif
  }

  /**
   * @brief Gather FFTW3 wisdom from all MPI processes to the root process.
   * @tparam PrecT Precision of the FFTW3 library.
   * @param comm MPI communicator.
   */
  template<typename PrecT>
  void gatherWisdom([[maybe_unused]] MPI_Comm comm)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::fftw3::hasMpiPrecision<typePrecision<PrecT>>)
    {
      detail::fftw3::MpiLib<typePrecision<PrecT>>::gatherWisdom(comm);
    }
# endif
  }
#endif
} // namespace mpst

  /// @brief Namespace alias for mpst namespace
  namespace mpi = mpst;
} // namespace afft

#endif /* AFFT_FFTW3_HPP */
