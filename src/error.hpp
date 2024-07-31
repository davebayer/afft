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

#include <afft/afft.hpp>

/**
 * @brief Handle exception.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error handleException(afft_ErrorDetails* errDetails) noexcept;

template<typename T = std::monostate>
void setErrorDetails(afft_ErrorDetails* errorDetails, const char* message, T&& retval = {})
{
  // Require that T fits into errorDetails->retval.
  static_assert(sizeof(T) <= sizeof(errorDetails->retval));

  if (errorDetails != nullptr)
  {
    // Set error message.
    std::strncpy(errorDetails->message, message, AFFT_MAX_ERROR_MESSAGE_SIZE);
    errorDetails->message[AFFT_MAX_ERROR_MESSAGE_SIZE - 1] = '\0';

    // Set return value.
    if constexpr (std::is_same_v<T, std::monostate>)
    {
      std::memset(&errorDetails->retval, 0, sizeof(errorDetails->retval));
    }
    else
    {
      std::memcpy(&errorDetails->retval, &retval, sizeof(retval));
    }
  }
}

/**
 * @brief Set error details message.
 * @param errDetails Error details.
 * @param message Message.
 */
void setErrorDetailsMessage(afft_ErrorDetails& errDetails, const char* message) noexcept;

/**
 * @brief Clear error details return value.
 * @param errDetails Error details.
 */
void clearErrorDetailsRetval(afft_ErrorDetails& errDetails) noexcept;

#endif /* ERROR_HPP */
