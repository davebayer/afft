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

#ifndef AFFT_VERSION_H
#ifndef AFFT_VERSION_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Version format string
#define AFFT_VERSION_FORMAT "%d.%d.%d"

/// @brief Version structure
struct afft_Version
{
  int major; ///< Major version number
  int minor; ///< Minor version number
  int patch; ///< Patch version number
};

/**
 * @brief Get the version of the AFFT library.
 * @return Version
 */
static inline afft_Version afft_getVersion()
{
  return (afft_Version){AFFT_VERSION_MAJOR, AFFT_VERSION_MINOR, AFFT_VERSION_PATCH};
}

/**
 * @brief Get the version of the clFFT library.
 * @return Version
 */
afft_Version afft_clfft_getVersion();

/**
 * @brief Get the version of the cuFFT library.
 * @return Version
 */
afft_Version afft_cufft_getVersion();

/**
 * @brief Get the version of the FFTW3 library.
 * @param precision Precision
 * @return Version
 */
afft_Version afft_fftw3_getVersion(afft_Precision precision);

/**
 * @brief Get the version of the HeFFTe library.
 * @return Version
 */
afft_Version afft_heffte_getVersion();

/**
 * @brief Get the version of the hipFFT library.
 * @return Version
 */
afft_Version afft_hipfft_getVersion();

/**
 * @brief Get the version of the MKL library.
 * @return Version
 */
afft_Version afft_mkl_getVersion();

/**
 * @brief Get the version of the PocketFFT library.
 * @return Version
 */
afft_Version afft_pocketfft_getVersion();

/**
 * @brief Get the version of the rocFFT library.
 * @return Version
 */
afft_Version afft_rocfft_getVersion();

/**
 * @brief Get the version of the VkFFT library.
 * @return Version
 */
afft_Version afft_vkfft_getVersion();

#ifdef __cplusplus
}
#endif

#endif /* AFFT_VERSION_H */
