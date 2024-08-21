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

#include "error.hpp"
#include "init.hpp"
#include "type.hpp"
#include "typeTraits.hpp"
#include "utils.hpp"
#include "detail/validate.hpp"
#ifdef AFFT_ENABLE_FFTW3
# include "detail/fftw3/Lib.hpp"
#endif

AFFT_EXPORT namespace afft::fftw3
{
  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param filename Name of the file to export the wisdom to.
   */
  void exportWisdomToFilename(Library library, const char* filename);

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param filename Name of the file to export the wisdom to.
   */
  void exportWisdomToFilename(Library library, std::string_view filename);

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param file File to export the wisdom to.
   */
  void exportWisdomToFile(Library library, FILE* file);

  /**
   * @brief Export FFTW3 wisdom to a string.
   * @param library Library to export the wisdom from.
   * @return String containing the wisdom. If the Library is not supported, an empty unique pointer is returned.
   */
  [[nodiscard]] std::unique_ptr<char[], FreeDeleter> exportWisdomToString(Library library);

  /**
   * @brief Export FFTW3 wisdom to a std::string.
   * @param library Library to export the wisdom from.
   * @return std::string containing the wisdom. If the Library is not supported, an empty string.
   */
  [[nodiscard]] std::string exportWisdomToStdString(Library library);

  /**
   * @brief Import FFTW3 wisdom from the system. Only on Unix and GNU systems.
   * @param library Library to import the wisdom to.
   */
  void importSystemWisdom(Library library);

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param library Library to import the wisdom to.
   * @param filename Name of the file to import the wisdom from.
   */
  void importWisdomFromFilename(Library library, const char* filename);

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param filename Name of the file to import the wisdom from.
   * @param library Library to import the wisdom to.
   */
  void importWisdomFromFilename(Library library, std::string_view filename);

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param library Library to import the wisdom to.
   * @param file File to import the wisdom from.
   */
  void importWisdomFromFile(Library library, FILE* file);

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @param library Library to import the wisdom to.
   * @param wisdom String containing the wisdom.
   */
  void importWisdomFromString(Library library, const char* wisdom);

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @param library Library to import the wisdom to.
   * @param wisdom String containing the wisdom.
   */
  void importWisdomFromString(Library library, std::string_view wisdom);

  /**
   * @brief Forget all FFTW3 wisdom.
   * @param library Library to forget the wisdom.
   */
  void forgetWisdom(Library library);

#ifdef AFFT_ENABLE_MPI
  namespace mpi
  {
    /**
     * @brief Broadcast FFTW3 wisdom to all MPI processes from the root process.
     * @param library Library to be the wisdom broadcasted for.
     * @param comm MPI communicator.
     */
    void broadcastWisdom(Library library, MPI_Comm comm);

    /**
     * @brief Gather FFTW3 wisdom from all MPI processes to the root process.
     * @param library Library to be the wisdom gathered for.
     * @param comm MPI communicator.
     */
    void gatherWisdom(Library library, MPI_Comm comm);
  } // namespace mpi
#endif
} // namespace afft::fftw3

#ifdef AFFT_HEADER_ONLY

AFFT_EXPORT namespace afft::fftw3
{
  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param filename Name of the file to export the wisdom to.
   */
  AFFT_HEADER_ONLY_INLINE void exportWisdomToFilename(Library library, [[maybe_unused]] const char* filename)
  {
    detail::validate(library);

    init();

    int retval{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      retval = detail::fftw3::Lib<Library::_float>::exportWisdomToFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      retval = detail::fftw3::Lib<Library::_double>::exportWisdomToFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      retval = detail::fftw3::Lib<Library::longDouble>::exportWisdomToFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      retval = detail::fftw3::Lib<Library::quad>::exportWisdomToFilename(filename);
      break;
# endif
    default:
      retval = 1;
      break;
    }

    if (retval == 0)
    {
      throw Exception{Error::fftw3, "failed to export wisdom to filename"};
    }
  }

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param filename Name of the file to export the wisdom to.
   */
  AFFT_HEADER_ONLY_INLINE void exportWisdomToFilename(Library library, std::string_view filename)
  {
    exportWisdomToFilename(library, filename.data());
  }

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @param library Library to export the wisdom from.
   * @param file File to export the wisdom to.
   */
  AFFT_HEADER_ONLY_INLINE void exportWisdomToFile(Library library, [[maybe_unused]] FILE* file)
  {
    detail::validate(library);

    init();

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      detail::fftw3::Lib<Library::_float>::exportWisdomToFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      detail::fftw3::Lib<Library::_double>::exportWisdomToFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      detail::fftw3::Lib<Library::longDouble>::exportWisdomToFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      detail::fftw3::Lib<Library::quad>::exportWisdomToFile(file);
      break;
# endif
    default:
      break;
    }
  }

  /**
   * @brief Export FFTW3 wisdom to a string.
   * @param library Library to export the wisdom from.
   * @return String containing the wisdom. If the Library is not supported, an empty unique pointer is returned.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<char[], FreeDeleter> exportWisdomToString(Library library)
  {
    detail::validate(library);

    init();

    std::unique_ptr<char[], FreeDeleter> wisdom{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      wisdom.reset(detail::fftw3::Lib<Library::_float>::exportWisdomToString());
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      wisdom.reset(detail::fftw3::Lib<Library::_double>::exportWisdomToString());
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      wisdom.reset(detail::fftw3::Lib<Library::longDouble>::exportWisdomToString());
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      wisdom.reset(detail::fftw3::Lib<Library::quad>::exportWisdomToString());
      break;
# endif
    default:
      return {};
    }

    if (wisdom == nullptr)
    {
      throw Exception{Error::fftw3, "failed to export wisdom to string"};
    }

    return wisdom;
  }

  /**
   * @brief Export FFTW3 wisdom to a std::string.
   * @param library Library to export the wisdom from.
   * @return std::string containing the wisdom. If the Library is not supported, an empty string.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::string exportWisdomToStdString(Library library)
  {
    auto wisdom = exportWisdomToString(library);

    std::string str{};

    if (wisdom != nullptr)
    {
      str = wisdom.get();
    }

    return str;
  }

  /**
   * @brief Import FFTW3 wisdom from the system. Only on Unix and GNU systems.
   * @param library Library to import the wisdom to.
   */
  AFFT_HEADER_ONLY_INLINE void importSystemWisdom(Library library)
  {
    detail::validate(library);

    init();

    int retval{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      retval = detail::fftw3::Lib<Library::_float>::importSystemWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      retval = detail::fftw3::Lib<Library::_double>::importSystemWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      retval = detail::fftw3::Lib<Library::longDouble>::importSystemWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      retval = detail::fftw3::Lib<Library::quad>::importSystemWisdom();
      break;
# endif
    default:
      retval = 1;
      break;
    }

    if (retval == 0)
    {
      throw Exception{Error::fftw3, "failed to import system wisdom"};
    }
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param library Library to import the wisdom to.
   * @param filename Name of the file to import the wisdom from.
   */
  AFFT_HEADER_ONLY_INLINE void importWisdomFromFilename(Library library, [[maybe_unused]] const char* filename)
  {
    detail::validate(library);

    init();

    int retval{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      retval = detail::fftw3::Lib<Library::_float>::importWisdomFromFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      retval = detail::fftw3::Lib<Library::_double>::importWisdomFromFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      retval = detail::fftw3::Lib<Library::longDouble>::importWisdomFromFilename(filename);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      retval = detail::fftw3::Lib<Library::quad>::importWisdomFromFilename(filename);
      break;
# endif
    default:
      retval = 1;
      break;
    }

    if (retval == 0)
    {
      throw Exception{Error::fftw3, "failed to import wisdom from filename"};
    }
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param filename Name of the file to import the wisdom from.
   * @param library Library to import the wisdom to.
   */
  AFFT_HEADER_ONLY_INLINE void importWisdomFromFilename(Library library, std::string_view filename)
  {
    importWisdomFromFilename(library, filename.data());
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @param library Library to import the wisdom to.
   * @param file File to import the wisdom from.
   */
  AFFT_HEADER_ONLY_INLINE void importWisdomFromFile(Library library, [[maybe_unused]] FILE* file)
  {
    detail::validate(library);

    init();

    int retval{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      retval = detail::fftw3::Lib<Library::_float>::importWisdomFromFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      retval = detail::fftw3::Lib<Library::_double>::importWisdomFromFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      retval = detail::fftw3::Lib<Library::longDouble>::importWisdomFromFile(file);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      retval = detail::fftw3::Lib<Library::quad>::importWisdomFromFile(file);
      break;
# endif
    default:
      retval = 1;
      break;
    }

    if (retval == 0)
    {
      throw Exception{Error::fftw3, "failed to import wisdom from file"};
    }
  }

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @param library Library to import the wisdom to.
   * @param wisdom String containing the wisdom.
   */
  AFFT_HEADER_ONLY_INLINE void importWisdomFromString(Library library, [[maybe_unused]] const char* wisdom)
  {
    detail::validate(library);

    init();

    int retval{};

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      retval = detail::fftw3::Lib<Library::_float>::importWisdomFromString(wisdom);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      retval = detail::fftw3::Lib<Library::_double>::importWisdomFromString(wisdom);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      retval = detail::fftw3::Lib<Library::longDouble>::importWisdomFromString(wisdom);
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      retval = detail::fftw3::Lib<Library::quad>::importWisdomFromString(wisdom);
      break;
# endif
    default:
      retval = 1;
      break;
    }

    if (retval == 0)
    {
      throw Exception{Error::fftw3, "failed to import wisdom from string"};
    }
  }

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @param library Library to import the wisdom to.
   * @param wisdom String containing the wisdom.
   */
  AFFT_HEADER_ONLY_INLINE void importWisdomFromString(Library library, std::string_view wisdom)
  {
    importWisdomFromString(library, wisdom.data());
  }

  /**
   * @brief Forget all FFTW3 wisdom.
   * @param library Library to forget the wisdom.
   */
  AFFT_HEADER_ONLY_INLINE void forgetWisdom(Library library)
  {
    detail::validate(library);

    init();

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_FLOAT)
    case Library::_float:
      detail::fftw3::Lib<Library::_float>::forgetWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_DOUBLE)
    case Library::_double:
      detail::fftw3::Lib<Library::_double>::forgetWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_LONG)
    case Library::longDouble:
      detail::fftw3::Lib<Library::longDouble>::forgetWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_QUAD)
    case Library::quad:
      detail::fftw3::Lib<Library::quad>::forgetWisdom();
      break;
# endif
    default:
      break;
    }
  }

#ifdef AFFT_ENABLE_MPI
  /**
   * @brief Broadcast FFTW3 wisdom to all MPI processes from the root process.
   * @param library Library to be the wisdom broadcasted for.
   * @param comm MPI communicator.
   */
  AFFT_HEADER_ONLY_INLINE void mpi::broadcastWisdom(Library library, [[maybe_unused]] MPI_Comm comm)
  {
    detail::validate(library);

    init();

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_FLOAT)
    case Library::_float:
      detail::fftw3::Lib<Library::_float>::mpiBroadcastWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_DOUBLE)
    case Library::_double:
      detail::fftw3::Lib<Library::_double>::mpiBroadcastWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_LONG)
    case Library::longDouble:
      detail::fftw3::Lib<Library::longDouble>::mpiBroadcastWisdom();
      break;
# endif
    default:
      break;
    }
  }

  /**
   * @brief Gather FFTW3 wisdom from all MPI processes to the root process.
   * @param library Library to be the wisdom gathered for.
   * @param comm MPI communicator.
   */
  AFFT_HEADER_ONLY_INLINE void mpi::gatherWisdom(Library library, [[maybe_unused]] MPI_Comm comm)
  {
    detail::validate(library);

    init();

    switch (library)
    {
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_FLOAT)
    case Library::_float:
      detail::fftw3::Lib<Library::_float>::mpiGatherWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_DOUBLE)
    case Library::_double:
      detail::fftw3::Lib<Library::_double>::mpiGatherWisdom();
      break;
# endif
# if defined(AFFT_ENABLE_FFTW3) && defined(AFFT_FFTW3_HAS_MPI_LONG)
    case Library::longDouble:
      detail::fftw3::Lib<Library::longDouble>::mpiGatherWisdom();
      break;
# endif
    default:
      break;
    }
  }
#endif
} // namespace afft::fftw3

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_FFTW3_HPP */
