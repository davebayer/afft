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

#include "afft/afft.hpp"

/**
 * @brief Get the version of the clFFT library.
 * @return Version
 */
extern "C" afft_Version afft_clfft_getVersion()
{
  return afft::clfft::getVersion();
}

/**
 * @brief Get the version of the cuFFT library.
 * @return Version
 */
extern "C" afft_Version afft_cufft_getVersion()
{
  return afft::cufft::getVersion();
}

/**
 * @brief Get the version of the FFTW3 library.
 * @param precision Precision
 * @return Version
 */
extern "C" afft_Version afft_fftw3_getVersion(afft_Precision precision)
try
{
  return afft::fftw3::getVersion(static_cast<afft::Precision>(precision));
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
  return afft::heffte::getVersion();
}

/**
 * @brief Get the version of the hipFFT library.
 * @return Version
 */
extern "C" afft_Version afft_hipfft_getVersion()
{
  return afft::hipfft::getVersion();
}

/**
 * @brief Get the version of the MKL library.
 * @return Version
 */
extern "C" afft_Version afft_mkl_getVersion()
{
  return afft::mkl::getVersion();
}

/**
 * @brief Get the version of the PocketFFT library.
 * @return Version
 */
extern "C" afft_Version afft_pocketfft_getVersion()
{
  return afft::pocketfft::getVersion();
}

/**
 * @brief Get the version of the rocFFT library.
 * @return Version
 */
extern "C" afft_Version afft_rocfft_getVersion()
{
  return afft::rocfft::getVersion();
}

/**
 * @brief Get the version of the VkFFT library.
 * @return Version
 */
extern "C" afft_Version afft_vkfft_getVersion()
{
  return afft::vkfft::getVersion();
}
