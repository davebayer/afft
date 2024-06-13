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

#include "detail/utils.hpp"

namespace afft
{
  /**
   * @struct Version
   * @brief Version
   */
  struct Version
  {
    int major{}; ///< major version
    int minor{}; ///< minor version
    int patch{}; ///< patch version
  };

  inline std::string toString(const Version &version)
  {
    return detail::cformat("%d.%d.%d", version.major, version.minor, version.patch);
  }

  /**
   * @brief Get the version of the AFFT library.
   * @return Version
   */
  constexpr Version getVersion() noexcept
  {
    return {AFFT_VERSION_MAJOR, AFFT_VERSION_MINOR, AFFT_VERSION_PATCH};
  }

  namespace clfft
  {
    /**
     * @brief Get the version of the clFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace clfft

  namespace cufft
  {
    /**
     * @brief Get the version of the cuFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace cufft

  namespace fftw3
  {
    /**
     * @brief Get the version of the FFTW library.
     * @return Version
     */
    Version getVersion(Precision precision = Precision::_double) noexcept;
  } // namespace fftw3

  namespace heffte
  {
    /**
     * @brief Get the version of the HeFFTe library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace heffte

  namespace hipfft
  {
    /**
     * @brief Get the version of the hipFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace hipfft

  namespace mkl
  {
    /**
     * @brief Get the version of the MKL library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace mkl

  namespace pocketfft
  {
    /**
     * @brief Get the version of the PocketFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace pocketfft

  namespace rocfft
  {
    /**
     * @brief Get the version of the rocFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace rocfft

  namespace vkfft
  {
    /**
     * @brief Get the version of the VkFFT library.
     * @return Version
     */
    Version getVersion() noexcept;
  } // namespace vkfft
} // namespace afft

#ifdef AFFT_HEADER_ONLY

namespace afft
{
  /**
   * @brief Get the version of the clFFT library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version clfft::getVersion() noexcept
  {
# if AFFT_GPU_BACKEND_IS(OPENCL) && AFFT_BACKEND_IS_ENABLED(CLFFT)
    return {clfftVersionMajor, clfftVersionMinor, clfftVersionPatch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the cuFFT library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version cufft::getVersion() noexcept
  {
# if AFFT_GPU_BACKEND_IS(CUDA) && AFFT_BACKEND_IS_ENABLED(CUFFT)
    return {CUFFT_VERSION / 1000, (CUFFT_VERSION % 1000) / 10, CUFFT_VERSION % 10};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the FFTW library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version fftw3::getVersion([[maybe_unused]] Precision precision) noexcept
  {
    Version version{};

# if AFFT_BACKEND_IS_ENABLED(FFTW3)
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
      std::sscanf(version, "%d.%d.%d", &version.major, &version.minor, &version.patch);
    }
# endif
    
    return version;
  }

  /**
   * @brief Get the version of the HeFFTe library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version heffte::getVersion() noexcept
  {
# if AFFT_MP_BACKEND_IS(MPI) && AFFT_BACKEND_IS_ENABLED(HEFFTE)
    return {Heffte_VERSION_MAJOR, Heffte_VERSION_MINOR, Heffte_VERSION_PATCH};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the hipFFT library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version hipfft::getVersion() noexcept
  {
# if AFFT_GPU_BACKEND_IS(HIP) && AFFT_BACKEND_IS_ENABLED(HIPFFT)
    return {hipfftVersionMajor, hipfftVersionMinor, hipfftVersionPatch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the MKL library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version mkl::getVersion() noexcept
  {
# if AFFT_BACKEND_IS_ENABLED(MKL)
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
  AFFT_HEADER_ONLY_INLINE Version pocketfft::getVersion() noexcept
  {
    return {}; // PocketFFT does not provide a version
  }

  /**
   * @brief Get the version of the rocFFT library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version rocfft::getVersion() noexcept
  {
# if AFFT_GPU_BACKEND_IS(HIP) && AFFT_BACKEND_IS_ENABLED(ROCFFT)
    return {rocfft_version_major, rocfft_version_minor, rocfft_version_patch};
# else
    return {};
# endif
  }

  /**
   * @brief Get the version of the VkFFT library.
   * @return Version
   */
  AFFT_HEADER_ONLY_INLINE Version vkfft::getVersion() noexcept
  {
# if AFFT_BACKEND_IS_ENABLED(VKFFT)
    const int version = VkFFTGetVersion();

    return {version / 10000, (version % 10000) / 100, version % 100};
# else
    return {};
# endif
  }
} // namespace afft

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_VERSION_HPP */
