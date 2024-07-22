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

#ifndef AFFT_DETAIL_HIPFFT_ERROR_HPP
#define AFFT_DETAIL_HIPFFT_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../error.hpp"

namespace afft::detail::hipfft
{
  /**
   * @brief Check if hipFFT result is ok.
   * @param result hipFFT result.
   * @return True if result is HIPFFT_SUCCESS, false otherwise.
   */
  [[nodiscard]] inline constexpr bool isOk(hipfftResult result)
  {
    return (result == HIPFFT_SUCCESS);
  }

  /**
   * @brief Check if hipFFT result is valid.
   * @param result hipFFT result.
   * @throw BackendError if result is not valid.
   */
  inline void checkError(hipfftResult result)
  {
    auto getErrorMsg = [](hipfftResult result) constexpr
    {
      switch (result)
      {
      case HIPFFT_SUCCESS:
        return "hipFFT operation was successful";
      case HIPFFT_INVALID_PLAN:
        return "invalid plan handle";
      case HIPFFT_ALLOC_FAILED:
        return "failed to allocate GPU or CPU memory";
      case HIPFFT_INVALID_TYPE:
        return "invalid type";
      case HIPFFT_INVALID_VALUE:
        return "invalid pointer or parameter";
      case HIPFFT_INTERNAL_ERROR:
        return "driver or internal library error";
      case HIPFFT_EXEC_FAILED:
        return "failed to execute";
      case HIPFFT_SETUP_FAILED:
        return "failed to initialize";
      case HIPFFT_INVALID_SIZE:
        return "invalid transform size";
      case HIPFFT_UNALIGNED_DATA:
        return "unaligned data";
      case HIPFFT_INCOMPLETE_PARAMETER_LIST:
        return "missing parameters in call";
      case HIPFFT_INVALID_DEVICE:
        return "invalid device";
      case HIPFFT_PARSE_ERROR:
        return "internal plan database error";
      case HIPFFT_NO_WORKSPACE:
        return "no workspace";
      case HIPFFT_NOT_IMPLEMENTED:
        return "unimplemented feature";
      case HIPFFT_NOT_SUPPORTED:
        return "operation is not supported for parameters given";
      default:
        return "unknown error";
      }
    };

    if (!isOk(result))
    {
      throw Exception{Error::hipfft, getErrorMsg(result)};
    }
  }
} // namespace afft::detail::hipfft

#endif /* AFFT_DETAIL_HIPFFT_ERROR_HPP */
