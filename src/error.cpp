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

#include "error.hpp"

/**
 * @brief Set error details message.
 * @param errDetails Error details.
 * @param message Message.
 */
static inline void setErrorDetailsMessage(afft_ErrorDetails& errDetails, const char* message) noexcept
{
  std::strncpy(errDetails.message, message, AFFT_MAX_ERROR_MESSAGE_SIZE);
  errDetails.message[AFFT_MAX_ERROR_MESSAGE_SIZE - 1] = '\0';
}

/**
 * @brief Clear error details return value.
 * @param errDetails Error details.
 */
static inline void clearErrorDetailsRetval(afft_ErrorDetails& errDetails) noexcept
{
  std::memset(&errDetails.retval, 0, sizeof(errDetails.retval));
}

/**
 * @brief Handle exception.
 * @param e Exception.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error handleException(const afft::Exception& e, afft_ErrorDetails* errDetails) noexcept
{
  if (errDetails != nullptr)
  {
    setErrorDetailsMessage(*errDetails, e.what());

    auto setRetval = [&](auto& member) noexcept -> void
    {
      using T = std::decay_t<decltype(member)>;

      const T* value = std::get_if<T>(&e.getErrorRetval());

      if (value != nullptr)
      {
        member = *value;
      }
      else
      {
        clearErrorDetailsRetval(*errDetails);
      }
    };

    switch (e.getError())
    {
#   ifdef AFFT_ENABLE_MPI
    case afft::Error::mpi:
      setRetval(errDetails->retval.mpi);
      break;
#   endif
#   ifdef AFFT_ENABLE_CUDA
    case afft::Error::cudaDriver:
      setRetval(errDetails->retval.cudaDriver);
      break;
    case afft::Error::cudaRuntime:
      setRetval(errDetails->retval.cudaRuntime);
      break;
    case afft::Error::cudaRtc:
      setRetval(errDetails->retval.cudaRtc);
      break;
#   endif
#   ifdef AFFT_ENABLE_HIP
    case afft::Error::hip:
      setRetval(errDetails->retval.hip);
      break;
#   endif
#   ifdef AFFT_ENABLE_OPENCL
    case afft::Error::opencl:
      setRetval(errDetails->retval.opencl);
      break;
#   endif
    default:
      clearErrorDetailsRetval(*errDetails);
      break;
    }
  }

  return Convert<afft::Error>::toC(e.getError());
}

/**
 * @brief Handle exception.
 * @param e Exception.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error handleException(const std::exception& e, afft_ErrorDetails* errDetails) noexcept
{
  if (errDetails != nullptr)
  {
    setErrorDetailsMessage(*errDetails, e.what());
    clearErrorDetailsRetval(*errDetails);
  }

  return afft_Error_internal;
}
