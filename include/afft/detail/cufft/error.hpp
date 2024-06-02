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

#ifndef AFFT_DETAIL_CUFFT_ERROR_HPP
#define AFFT_DETAIL_CUFFT_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../exception.hpp"

namespace afft::detail::cufft
{
  /**
   * @brief Check if cuFFT result is ok.
   * @param result cuFFT result.
   * @return True if result is ok, false otherwise.
   */
  [[nodiscard]] inline constexpr bool isOk(cufftResult result)
  {
    return (result == CUFFT_SUCCESS);
  }

  /**
   * @brief Check if cuFFT result is valid.
   * @param result cuFFT result.
   * @throw BackendError if result is not valid.
   */
  inline void checkError(cufftResult result)
  {
    auto getErrorMsg = [](cufftResult result) constexpr -> std::string_view
    {
      switch (result)
      {
      case CUFFT_SUCCESS:
        return "no error";
      case CUFFT_INVALID_PLAN:
        return "invalid plan handle";
      case CUFFT_ALLOC_FAILED:
        return "allocation of memory failed";
      case CUFFT_INVALID_TYPE:
        return "invalid type";
      case CUFFT_INVALID_VALUE:
        return "invalid value";
      case CUFFT_INTERNAL_ERROR:
        return "internal error";
      case CUFFT_EXEC_FAILED:
        return "plan execution failed";
      case CUFFT_SETUP_FAILED:
        return "setup failed";
      case CUFFT_INVALID_SIZE:
        return "invalid size";
      case CUFFT_UNALIGNED_DATA:
        return "invalid data alignment";
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "incomplete parameter list";
      case CUFFT_INVALID_DEVICE:
        return "invalid device";
      case CUFFT_PARSE_ERROR:
        return "parse error";
      case CUFFT_NO_WORKSPACE:
        return "no workspace";
      case CUFFT_NOT_IMPLEMENTED:
        return "unimplemented feature";
      case CUFFT_LICENSE_ERROR:
        return "license error";
      case CUFFT_NOT_SUPPORTED:
        return "unsupported functionality";
      default:
        return "unknown error";
      }
    };

    if (!isOk(result))
    {
      throw makeException<BackendError>(Backend::cufft, getErrorMsg(result));
    }
  }
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_ERROR_HPP */
