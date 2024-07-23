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

#ifndef ERROR_HPP
#define ERROR_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

/// @brief Specialization of Convert for afft::Error. 
template<>
struct Convert<afft::Error>
  : EnumConvertBase<afft::Error, afft_Error>
{
  static_assert(cmpEnumValues(afft::Error::internal,        afft_Error_internal));
  static_assert(cmpEnumValues(afft::Error::invalidArgument, afft_Error_invalidArgument));
  static_assert(cmpEnumValues(afft::Error::mpi,             afft_Error_mpi));
  static_assert(cmpEnumValues(afft::Error::cudaDriver,      afft_Error_cudaDriver));
  static_assert(cmpEnumValues(afft::Error::cudaRuntime,     afft_Error_cudaRuntime));
  static_assert(cmpEnumValues(afft::Error::cudaRtc,         afft_Error_cudaRtc));
  static_assert(cmpEnumValues(afft::Error::hip,             afft_Error_hip));
  static_assert(cmpEnumValues(afft::Error::opencl,          afft_Error_opencl));
  static_assert(cmpEnumValues(afft::Error::clfft,           afft_Error_clfft));
  static_assert(cmpEnumValues(afft::Error::cufft,           afft_Error_cufft));
  static_assert(cmpEnumValues(afft::Error::fftw3,           afft_Error_fftw3));
  static_assert(cmpEnumValues(afft::Error::heffte,          afft_Error_heffte));
  static_assert(cmpEnumValues(afft::Error::hipfft,          afft_Error_hipfft));
  static_assert(cmpEnumValues(afft::Error::mkl,             afft_Error_mkl));
  static_assert(cmpEnumValues(afft::Error::pocketfft,       afft_Error_pocketfft));
  static_assert(cmpEnumValues(afft::Error::rocfft,          afft_Error_rocfft));
  static_assert(cmpEnumValues(afft::Error::vkfft,           afft_Error_vkfft));
};

/**
 * @brief Handle exception.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error handleException(afft_ErrorDetails* errDetails) noexcept;

#endif /* ERROR_HPP */
