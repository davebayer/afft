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

#ifndef AFFT_DETAIL_CLFFT_ERROR_HPP
#define AFFT_DETAIL_CLFFT_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../error.hpp"

namespace afft::detail::clfft
{
  /**
   * @brief Check if clFFT status is ok.
   * @param result clFFT status.
   * @return True if result is CLFFT_SUCCESS, false otherwise.
   */
  [[nodiscard]] inline constexpr bool isOk(clfftStatus result)
  {
    return (result == CLFFT_SUCCESS);
  }

  /**
   * @brief Check if clFFT status is valid.
   * @param result clFFT status.
   * @throw BackendError if result is not valid.
   */
  inline void checkError(clfftStatus result)
  {
    auto getErrorMsg = [](clfftStatus result)
    {
      switch (result)
      {
      case CLFFT_INVALID_GLOBAL_WORK_SIZE:
      case CLFFT_INVALID_MIP_LEVEL:
      case CLFFT_INVALID_BUFFER_SIZE:
      case CLFFT_INVALID_GL_OBJECT:
      case CLFFT_INVALID_OPERATION:
      case CLFFT_INVALID_EVENT:
      case CLFFT_INVALID_EVENT_WAIT_LIST:
      case CLFFT_INVALID_GLOBAL_OFFSET:
      case CLFFT_INVALID_WORK_ITEM_SIZE:
      case CLFFT_INVALID_WORK_GROUP_SIZE:
      case CLFFT_INVALID_WORK_DIMENSION:
      case CLFFT_INVALID_KERNEL_ARGS:
      case CLFFT_INVALID_ARG_SIZE:
      case CLFFT_INVALID_ARG_VALUE:
      case CLFFT_INVALID_ARG_INDEX:
      case CLFFT_INVALID_KERNEL:
      case CLFFT_INVALID_KERNEL_DEFINITION:
      case CLFFT_INVALID_KERNEL_NAME:
      case CLFFT_INVALID_PROGRAM_EXECUTABLE:
      case CLFFT_INVALID_PROGRAM:
      case CLFFT_INVALID_BUILD_OPTIONS:
      case CLFFT_INVALID_BINARY:
      case CLFFT_INVALID_SAMPLER:
      case CLFFT_INVALID_IMAGE_SIZE:
      case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      case CLFFT_INVALID_MEM_OBJECT:
      case CLFFT_INVALID_HOST_PTR:
      case CLFFT_INVALID_COMMAND_QUEUE:
      case CLFFT_INVALID_QUEUE_PROPERTIES:
      case CLFFT_INVALID_CONTEXT:
      case CLFFT_INVALID_DEVICE:
      case CLFFT_INVALID_PLATFORM:
      case CLFFT_INVALID_DEVICE_TYPE:
      case CLFFT_INVALID_VALUE:
      case CLFFT_MAP_FAILURE:
      case CLFFT_BUILD_PROGRAM_FAILURE:
      case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
      case CLFFT_IMAGE_FORMAT_MISMATCH:
      case CLFFT_MEM_COPY_OVERLAP:
      case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
      case CLFFT_OUT_OF_HOST_MEMORY:
      case CLFFT_OUT_OF_RESOURCES:
      case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
      case CLFFT_COMPILER_NOT_AVAILABLE:
      case CLFFT_DEVICE_NOT_AVAILABLE:
      case CLFFT_DEVICE_NOT_FOUND:
      case CLFFT_SUCCESS:
        
        break;
      case CLFFT_NOTIMPLEMENTED:
      case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
      case CLFFT_FILE_NOT_FOUND:
      case CLFFT_FILE_CREATE_FAILURE:
      case CLFFT_VERSION_MISMATCH:
      case CLFFT_INVALID_PLAN:
      case CLFFT_DEVICE_NO_DOUBLE:
      case CLFFT_DEVICE_MISMATCH:

      case CLFFT_BUGCHECK:
      default:
        return "unknown error";
      }
    };

    if (!isOk(result))
    {
      throw Exception{Error::clfft, getErrorMsg(result)};
    }
  }
} // namespace afft::detail::clfft

#endif /* AFFT_DETAIL_CLFFT_ERROR_HPP */
