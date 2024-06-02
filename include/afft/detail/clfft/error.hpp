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

#include "../../exception.hpp"

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
        return "invalid global work size";
      case CLFFT_INVALID_MIP_LEVEL:
        return "invalid mip level";
      case CLFFT_INVALID_BUFFER_SIZE:
        return "invalid buffer size";
      case CLFFT_INVALID_GL_OBJECT:
        return "invalid gl object";
      case CLFFT_INVALID_OPERATION:
        return "invalid operation";
      case CLFFT_INVALID_EVENT:
        return "invalid event";
      case CLFFT_INVALID_EVENT_WAIT_LIST:
        return "invalid even wait list";
      case CLFFT_INVALID_GLOBAL_OFFSET:
        return "invalid global offset";
      case CLFFT_INVALID_WORK_ITEM_SIZE:
        return "invalid work item size";
      case CLFFT_INVALID_WORK_GROUP_SIZE:
        return "invalid work group size";
      case CLFFT_INVALID_WORK_DIMENSION:
        return "invalid work dimension";
      case CLFFT_INVALID_KERNEL_ARGS:
        return "invalid kernel argumentss";
      case CLFFT_INVALID_ARG_SIZE:
        return "invalid argument size";
      case CLFFT_INVALID_ARG_VALUE:
        return "invalid argument value";
      case CLFFT_INVALID_ARG_INDEX:
        return "invalid argument index";
      case CLFFT_INVALID_KERNEL:
        return "invalid kernel";
      case CLFFT_INVALID_KERNEL_DEFINITION:
        return "invalid kernel definition";
      case CLFFT_INVALID_KERNEL_NAME:
        return "invalid kernel name";
      case CLFFT_INVALID_PROGRAM_EXECUTABLE:
        return "invalid program executable";
      case CLFFT_INVALID_PROGRAM:
        return "invalid program";
      case CLFFT_INVALID_BUILD_OPTIONS:
        return "invalid build options";
      case CLFFT_INVALID_BINARY:
        return "invalid binary";
      case CLFFT_INVALID_SAMPLER:
        return "invalid sampler";
      case CLFFT_INVALID_IMAGE_SIZE:
        return "invalid image size";
      case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "invalid image format description";
      case CLFFT_INVALID_MEM_OBJECT:
        return "invalid memory object";
      case CLFFT_INVALID_HOST_PTR:
        return "invalid host pointer";
      case CLFFT_INVALID_COMMAND_QUEUE:
        return "invalid command queue";
      case CLFFT_INVALID_QUEUE_PROPERTIES:
        return "invalid queue properties";
      case CLFFT_INVALID_CONTEXT:
        return "invalid context";
      case CLFFT_INVALID_DEVICE:
        return "invalid device";
      case CLFFT_INVALID_PLATFORM:
        return "invalid platform";
      case CLFFT_INVALID_DEVICE_TYPE:
        return "invalid device type";
      case CLFFT_INVALID_VALUE:
        return "invalid value";
      case CLFFT_MAP_FAILURE:
        return "map failure";
      case CLFFT_BUILD_PROGRAM_FAILURE:
        return "failed to build program";
      case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
        return "unsupported image format";
      case CLFFT_IMAGE_FORMAT_MISMATCH:
        return "image format mismatch";
      case CLFFT_MEM_COPY_OVERLAP:
        return "overlap of memory copy";
      case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
        return "unavailable profiling information";
      case CLFFT_OUT_OF_HOST_MEMORY:
        return "out of host memory";
      case CLFFT_OUT_OF_RESOURCES:
        return "out of resources";
      case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
        return "memory object allocation failed";
      case CLFFT_COMPILER_NOT_AVAILABLE:
        return "unavailable compiler";
      case CLFFT_DEVICE_NOT_AVAILABLE:
        return "unavailable device";
      case CLFFT_DEVICE_NOT_FOUND:
        return "device not found";
      case CLFFT_SUCCESS:
        return "no error";
      case CLFFT_BUGCHECK:
        return "bugcheck";
      case CLFFT_NOTIMPLEMENTED:
        return "unimplemented feature";
      case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
        return "unimplemented transposed";
      case CLFFT_FILE_NOT_FOUND:
        return "file not found";
      case CLFFT_FILE_CREATE_FAILURE:
        return "failed to create file";
      case CLFFT_VERSION_MISMATCH:
        return "version mismatch";
      case CLFFT_INVALID_PLAN:
        return "invalid plan";
      case CLFFT_DEVICE_NO_DOUBLE:
        return "device does not support double precision";
      case CLFFT_DEVICE_MISMATCH:
        return "device mismatch";
      default:
        return "unknown error";
      }
    };

    if (!isOk(result))
    {
      throw BackendError{Backend::clfft, getErrorMsg(result)};
    }
  }
} // namespace afft::detail::clfft

#endif /* AFFT_DETAIL_CLFFT_ERROR_HPP */
