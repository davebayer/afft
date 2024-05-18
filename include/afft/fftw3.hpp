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

#include "detail/fftw3/Lib.hpp"

AFFT_EXPORT namespace afft::fftw3
{
  /**
   * @brief Export FFTW3 wisdom to a file.
   * @tparam prec Precision of the FFTW3 library.
   * @param filename Name of the file to export the wisdom to.
   */
  template<Precision prec>
  void exportWisdomToFilename([[maybe_unused]] std::string_view filename)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::exportWisdomToFilename(filename.data()))
      {
        throw std::runtime_error("Failed to export FFTW3 wisdom to file");
      }
    }
# endif
  }

  /**
   * @brief Export FFTW3 wisdom to a file.
   * @tparam prec Precision of the FFTW3 library.
   * @param file File to export the wisdom to.
   */
  template<Precision prec>
  void exportWisdomToFile([[maybe_unused]] FILE* file)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::exportWisdomToFile(file))
      {
        throw std::runtime_error("Failed to export FFTW3 wisdom to file");
      }
    }
# endif
  }

  /**
   * @brief Export FFTW3 wisdom to a string.
   * @tparam prec Precision of the FFTW3 library.
   * @return String containing the wisdom.
   */
  template<Precision prec>
  [[nodiscard]] std::string exportWisdom()
  {
    std::string wisdom{};

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      struct FreeDeleter
      {
        void operator()(char* ptr) const
        {
          free(ptr);
        }
      };

      std::unique_ptr<char, FreeDeleter> orgWisdom{detail::fftw3::Lib<prec>::exportWisdomToString()};

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
   * @tparam prec Precision of the FFTW3 library.
   */
  template<Precision prec>
  void importSystemWisdom()
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::importSystemWisdom())
      {
        throw std::runtime_error("Failed to import FFTW3 system wisdom");
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @tparam prec Precision of the FFTW3 library.
   * @param filename Name of the file to import the wisdom from.
   */
  template<Precision prec>
  void importWisdomFromFilename([[maybe_unused]] std::string_view filename)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::importWisdomFromFilename(filename.data()))
      {
        throw std::runtime_error("Failed to import FFTW3 wisdom from file");
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a file.
   * @tparam prec Precision of the FFTW3 library.
   * @param file File to import the wisdom from.
   */
  template<Precision prec>
  void importWisdomFromFile([[maybe_unused]] FILE* file)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::importWisdomFromFile(file))
      {
        throw std::runtime_error("Failed to import FFTW3 wisdom from file");
      }
    }
# endif
  }

  /**
   * @brief Import FFTW3 wisdom from a string.
   * @tparam prec Precision of the FFTW3 library.
   * @param wisdom String containing the wisdom.
   */
  template<Precision prec>
  void importWisdom([[maybe_unused]] std::string_view wisdom)
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      if (!detail::fftw3::Lib<prec>::importWisdomFromString(wisdom.data()))
      {
        throw std::runtime_error("Failed to import FFTW3 wisdom");
      }
    }
# endif
  }

  /**
   * @brief Forget all FFTW3 wisdom.
   * @tparam prec Precision of the FFTW3 library.
   */
  template<Precision prec>
  void forgetWisdom()
  {
# if AFFT_BACKEND_IS_ENABLED(FFTW3)
    if constexpr (detail::hasPrecision<prec>())
    {
      detail::fftw3::Lib<prec>::forgetWisdom();
    }
# endif
  }
} // namespace afft

#endif /* AFFT_FFTW3_HPP */
