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

#ifndef AFFT_DETAIL_GPU_CUFFT_ERROR_HPP
#define AFFT_DETAIL_GPU_CUFFT_ERROR_HPP

#include <string>

#include <cufft.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for cufftResult.
 * @param result cuFFT result.
 * @return True if result is CUFFT_SUCCESS, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(cufftResult result)
{
  return (result == CUFFT_SUCCESS);
}

/**
 * @brief Specialization of makeErrorMessage method for cufftResult.
 * @param result cuFFT result.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(cufftResult result)
{
  auto get = [=]()
  {
    switch (result)
    {
    case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:           return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:             return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:             return "CUFFT_NOT_SUPPORTED";
    default:                              return "Unknown error";
    }
  };

  return format("[cuFFT error] {}", get());
}

#endif /* AFFT_DETAIL_GPU_CUFFT_ERROR_HPP */
