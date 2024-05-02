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

#ifndef AFFT_DETAIL_GPU_HIPFFT_ERROR_HPP
#define AFFT_DETAIL_GPU_HIPFFT_ERROR_HPP

#include <string>

#include <hipfft/hipfft.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for hipfftResult.
 * @param result hipFFT result.
 * @return True if result is HIPFFT_SUCCESS, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(hipfftResult result)
{
  return (result == HIPFFT_SUCCESS);
}

/**
 * @brief Specialization of makeErrorMessage method for hipfftResult.
 * @param result hipFFT result.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(hipfftResult result)
{
  auto get = [=]()
  {
    switch (result)
    {
    case HIPFFT_SUCCESS:                   return "hipFFT operation was successful";
    case HIPFFT_INVALID_PLAN:              return "hipFFT was passed an invalid plan handle";
    case HIPFFT_ALLOC_FAILED:              return "hipFFT failed to allocate GPU or CPU memory";
    case HIPFFT_INVALID_TYPE:              return "Invalid type";
    case HIPFFT_INVALID_VALUE:             return "User specified an invalid pointer or parameter";
    case HIPFFT_INTERNAL_ERROR:            return "Driver or internal hipFFT library error";
    case HIPFFT_EXEC_FAILED:               return "Failed to execute an FFT on the GPU";
    case HIPFFT_SETUP_FAILED:              return "hipFFT failed to initialize";
    case HIPFFT_INVALID_SIZE:              return "User specified an invalid transform size";
    case HIPFFT_UNALIGNED_DATA:            return "Unaligned data";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST: return "Missing parameters in call";
    case HIPFFT_INVALID_DEVICE:            return "Execution of a plan was on different GPU than plan creation";
    case HIPFFT_PARSE_ERROR:               return "Internal plan database error";
    case HIPFFT_NO_WORKSPACE:              return "No workspace has been provided prior to plan execution";
    case HIPFFT_NOT_IMPLEMENTED:           return "Function does not implement functionality for parameters given";
    case HIPFFT_NOT_SUPPORTED:             return "Operation is not supported for parameters given";
    default:                               return "Unknown error";
    }
  };
  
  return cformat("[hipFFT error] %s", get());
}

#endif /* AFFT_DETAIL_GPU_HIPFFT_ERROR_HPP */
