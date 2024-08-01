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

#ifndef AFFT_VERSION_HPP
#define AFFT_VERSION_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "type.hpp"
#include "detail/utils.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Version structure
  using Version = afft_Version;

  /**
   * @brief Convert a version to a string.
   * @param version Version
   * @return std::string
   */
  [[nodiscard]] inline std::string toString(const Version &version)
  {
    return detail::cformat(AFFT_VERSION_FORMAT, version.major, version.minor, version.patch);
  }

  /**
   * @brief Get the version of the AFFT library.
   * @return Version
   */
  [[nodiscard]] constexpr Version getVersion() noexcept
  {
    return {AFFT_VERSION_MAJOR, AFFT_VERSION_MINOR, AFFT_VERSION_PATCH};
  }

  namespace clfft
  {
    /**
     * @brief Get the version of the clFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace clfft

  namespace cufft
  {
    /**
     * @brief Get the version of the cuFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace cufft

  namespace fftw3
  {
    /**
     * @brief Get the version of the FFTW library.
     * @return Version
     */
    [[nodiscard]] Version getVersion(Precision precision = Precision::_double) noexcept;
  } // namespace fftw3

  namespace heffte
  {
    /**
     * @brief Get the version of the HeFFTe library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace heffte

  namespace hipfft
  {
    /**
     * @brief Get the version of the hipFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace hipfft

  namespace mkl
  {
    /**
     * @brief Get the version of the MKL library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace mkl

  namespace pocketfft
  {
    /**
     * @brief Get the version of the PocketFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace pocketfft

  namespace rocfft
  {
    /**
     * @brief Get the version of the rocFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace rocfft

  namespace vkfft
  {
    /**
     * @brief Get the version of the VkFFT library.
     * @return Version
     */
    [[nodiscard]] Version getVersion() noexcept;
  } // namespace vkfft
} // namespace afft

#ifdef AFFT_CXX_HAS_FORMAT
/// @brief Format a version.
AFFT_EXPORT template<>
struct std::formatter<afft::Version>
{
  /**
   * @brief Parse a format string.
   * @param ctx Format context
   * @return std::format_parse_context::iterator
   */
  [[nodiscard]] constexpr auto parse(std::format_parse_context& ctx) const noexcept
    -> std::format_parse_context::iterator
  {
    std::format_parse_context::iterator it{};

    for (it = ctx.begin(); it != ctx.end() && *it != '}'; ++it) {}

    return it;
  }

  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Version& version, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{}.{}.{}", version.major, version.minor, version.patch);
  }
};
#endif

#ifdef AFFT_HEADER_ONLY

AFFT_EXPORT namespace afft
{
  /**
   * @brief Get the version of the clFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version clfft::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_CLFFT
    return {clfftVersionMajor, clfftVersionMinor, clfftVersionPatch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the cuFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version cufft::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_CUFFT
    return {CUFFT_VER_MAJOR, CUFFT_VER_MINOR, CUFFT_VER_PATCH};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the FFTW library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version fftw3::getVersion([[maybe_unused]] Precision precision) noexcept
  {
    Version version{};

# ifdef AFFT_ENABLE_FFTW3
    const char* versionString{};

    switch (precision)
    {
#   ifdef AFFT_FFTW3_HAS_FLOAT
    case Precision::_float:
      versionString = fftwf_version;
      break;
#   endif
#   ifdef AFFT_FFTW3_HAS_DOUBLE
    case Precision::_double:
      versionString = fftw_version;
      break;
#   endif
#   ifdef AFFT_FFTW3_HAS_LONG_DOUBLE
    case Precision::_longDouble:
      versionString = fftwl_version;
      break;
#   endif
#   ifdef AFFT_FFTW3_HAS_QUAD
    case Precision::_quad:
      versionString = fftwq_version;
      break;
#   endif
    default:
      break;
    }

    if (version != nullptr)
    {
      if (std::sscanf(version, "fftw-%d.%d.%d", &version.major, &version.minor, &version.patch) != 3)
      {
        version = {};
      }
    }
# endif
    
    return version;
  }

  /**
   * @brief Get the version of the HeFFTe library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version heffte::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_HEFFTE
    return {Heffte_VERSION_MAJOR, Heffte_VERSION_MINOR, Heffte_VERSION_PATCH};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the hipFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version hipfft::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_HIPFFT
    return {hipfftVersionMajor, hipfftVersionMinor, hipfftVersionPatch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the MKL library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version mkl::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_MKL
    MKLVersion version;
    mkl_get_version(&version);
    return {version.MajorVersion, version.MinorVersion, version.UpdateVersion};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the PocketFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version pocketfft::getVersion() noexcept
  {
    return {}; // PocketFFT does not provide a version
  }

  /**
   * @brief Get the version of the rocFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version rocfft::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_ROCFFT
    return {rocfft_version_major, rocfft_version_minor, rocfft_version_patch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the VkFFT library.
   * @return Version
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE Version vkfft::getVersion() noexcept
  {
# ifdef AFFT_ENABLE_VKFFT
    const int version = VkFFTGetVersion();

    return {version / 10000, (version % 10000) / 100, version % 100};
# else
    return {};
# endif
  }
} // namespace afft

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_VERSION_HPP */
