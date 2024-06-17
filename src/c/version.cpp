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

#include "common.hpp"
#include "version.hpp"

/**
 * @brief Get the version of the clFFT library.
 * @return Version
 */
extern "C" afft_Version afft_clfft_getVersion()
{
  return Convert<afft::Version>::toC(afft::clfft::getVersion());
}

/**
 * @brief Get the version of the cuFFT library.
 * @return Version
 */
extern "C" afft_Version afft_cufft_getVersion()
{
  return Convert<afft::Version>::toC(afft::cufft::getVersion());
}

/**
 * @brief Get the version of the FFTW3 library.
 * @param precision Precision
 * @return Version
 */
extern "C" afft_Version afft_fftw3_getVersion(afft_Precision precision)
try
{
  return Convert<afft::Version>::toC(afft::fftw3::getVersion(Convert<afft::Precision>::fromC(precision)));
}
catch (...)
{
  return afft_Version{};
}

/**
 * @brief Get the version of the HeFFTe library.
 * @return Version
 */
extern "C" afft_Version afft_heffte_getVersion()
{
  return Convert<afft::Version>::toC(afft::heffte::getVersion());
}

/**
 * @brief Get the version of the hipFFT library.
 * @return Version
 */
extern "C" afft_Version afft_hipfft_getVersion()
{
  return Convert<afft::Version>::toC(afft::hipfft::getVersion());
}

/**
 * @brief Get the version of the MKL library.
 * @return Version
 */
extern "C" afft_Version afft_mkl_getVersion()
{
  return Convert<afft::Version>::toC(afft::mkl::getVersion());
}

/**
 * @brief Get the version of the PocketFFT library.
 * @return Version
 */
extern "C" afft_Version afft_pocketfft_getVersion()
{
  return Convert<afft::Version>::toC(afft::pocketfft::getVersion());
}

/**
 * @brief Get the version of the rocFFT library.
 * @return Version
 */
extern "C" afft_Version afft_rocfft_getVersion()
{
  return Convert<afft::Version>::toC(afft::rocfft::getVersion());
}

/**
 * @brief Get the version of the VkFFT library.
 * @return Version
 */
extern "C" afft_Version afft_vkfft_getVersion()
{
  return Convert<afft::Version>::toC(afft::vkfft::getVersion());
}
