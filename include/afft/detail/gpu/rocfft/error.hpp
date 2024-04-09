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

#ifndef AFFT_DETAIL_GPU_ROCFFT_ERROR_HPP
#define AFFT_DETAIL_GPU_ROCFFT_ERROR_HPP

#include <string>

#include <rocfft/rocfft.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for rocfft_status.
 * @param status rocFFT status.
 * @return True if status is rocfft_status_success, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(rocfft_status status)
{
  return (status == rocfft_status_success);
}

/**
 * @brief Specialization of makeErrorMessage method for rocfft_status.
 * @param status rocFFT status.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(rocfft_status status)
{
  auto get = [=]()
  {
    switch(status)
    {
    case rocfft_status_success:             return "rocfft_status_success";
    case rocfft_status_failure:             return "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:   return "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:  return "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:  return "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides:     return "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:    return "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset:      return "rocfft_status_invalid_offset";
    case rocfft_status_invalid_work_buffer: return "rocfft_status_invalid_work_buffer";
    default:                                return "Unknown error";
    }
  };

  return format("[rocFFT error] {}", get());
}

#endif /* AFFT_DETAIL_GPU_ROCFFT_ERROR_HPP */
