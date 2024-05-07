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

#ifndef AFFT_DETAIL_CPU_MKL_ERROR_HPP
#define AFFT_DETAIL_CPU_MKL_ERROR_HPP

#include <string>

#include <mkl_dfti.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for MKL_LONG.
 * @param result MKL error.
 * @return True if result is DFTI_NO_ERROR, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(MKL_LONG result)
{
  return (result == DFTI_NO_ERROR);
}

/**
 * @brief Specialization of makeErrorMessage method for MKL_LONG.
 * @param result MKL error.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(MKL_LONG result)
{
  auto get = [=]()
  {
    const char* msg = DftiErrorMessage(result);

    return (msg != nullptr) ? msg : "Unknown error";
  };

  return cformat("[MKL error] %s", get());
}

#endif /* AFFT_DETAIL_CPU_MKL_ERROR_HPP */
