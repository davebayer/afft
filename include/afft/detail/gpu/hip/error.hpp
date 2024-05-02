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

#ifndef AFFT_DETAIL_GPU_HIP_ERROR_HPP
#define AFFT_DETAIL_GPU_HIP_ERROR_HPP

#include <string>

#include <hip/hip_runtime.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for hipError_t.
 * @param error HIP error.
 * @return True if error is hipSuccess, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(hipError_t error)
{
  return (error == hipSuccess);
}

/**
 * @brief Specialization of makeErrorMessage method for hipError_t.
 * @param error HIP error.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(hipError_t error)
{
  return cformat("[HIP error] %s - %s", hipGetErrorName(error), hipGetErrorString(error));
}

#endif /* AFFT_DETAIL_GPU_HIP_ERROR_HPP */
