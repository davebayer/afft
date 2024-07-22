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

#ifndef AFFT_DETAIL_ROCFFT_ERROR_HPP
#define AFFT_DETAIL_ROCFFT_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../error.hpp"

namespace afft::detail::rocfft
{
  /**
   * @brief Check if rocFFT status is ok.
   * @param status rocFFT status.
   * @return True if status is rocfft_status_success, false otherwise.
   */
  [[nodiscard]] constexpr bool isOk(rocfft_status status)
  {
    return (status == rocfft_status_success);
  }

  /**
   * @brief Check if rocFFT status is valid.
   * @param status rocFFT status.
   * @throw BackendError if status is not valid.
   */
  constexpr void checkError(rocfft_status status)
  {
    auto getErrorMsg = [](rocfft_status status)
    {
      switch(status)
      {
      case rocfft_status_success:
        return "no error";
      case rocfft_status_failure:
        return "failure";
      case rocfft_status_invalid_arg_value:
        return "invalid argument value";
      case rocfft_status_invalid_dimensions:
        return "invalid dimensions";
      case rocfft_status_invalid_array_type:
        return "invalid array type";
      case rocfft_status_invalid_strides:
        return "invalid strides";
      case rocfft_status_invalid_distance:
        return "invalid distance";
      case rocfft_status_invalid_offset:
        return "invalid offset";
      case rocfft_status_invalid_work_buffer:
        return "invalid work buffer";
      default:
        return "unknown error";
      }
    };

    if (!isOk(status))
    {
      throw Exception{Error::rocfft, getErrorMsg(status)};
    }
  }
} // namespace afft::detail::rocfft

#endif /* AFFT_DETAIL_ROCFFT_ERROR_HPP */
