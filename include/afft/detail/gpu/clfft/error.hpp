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

#ifndef AFFT_DETAIL_GPU_CLFFT_ERROR_HPP
#define AFFT_DETAIL_GPU_CLFFT_ERROR_HPP

#include <clFFT.h>

#include "../../error.hpp"
#include "../../utils.hpp"

/**
 * @brief Specialization of isOk method for clfftStatus.
 * @param result clFFT result.
 * @return True if result is CLFFT_SUCCESS, false otherwise.
 */
template<>
[[nodiscard]] constexpr bool afft::detail::Error::isOk(clfftStatus result)
{
  return (result == CLFFT_SUCCESS);
}

/**
 * @brief Specialization of makeErrorMessage method for clfftStatus.
 * @param result clFFT result.
 * @return Error message.
 */
template<>
[[nodiscard]] std::string afft::detail::Error::makeErrorMessage(clfftStatus result)
{
  auto get = [=]()
  {
    switch (result)
    {
    case CLFFT_INVALID_GLOBAL_WORK_SIZE:        return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
    case CLFFT_INVALID_MIP_LEVEL:               return "CLFFT_INVALID_MIP_LEVEL";
    case CLFFT_INVALID_BUFFER_SIZE:             return "CLFFT_INVALID_BUFFER_SIZE";
    case CLFFT_INVALID_GL_OBJECT:               return "CLFFT_INVALID_GL_OBJECT";
    case CLFFT_INVALID_OPERATION:               return "CLFFT_INVALID_OPERATION";
    case CLFFT_INVALID_EVENT:                   return "CLFFT_INVALID_EVENT";
    case CLFFT_INVALID_EVENT_WAIT_LIST:         return "CLFFT_INVALID_EVENT_WAIT_LIST";
    case CLFFT_INVALID_GLOBAL_OFFSET:           return "CLFFT_INVALID_GLOBAL_OFFSET";
    case CLFFT_INVALID_WORK_ITEM_SIZE:          return "CLFFT_INVALID_WORK_ITEM_SIZE";
    case CLFFT_INVALID_WORK_GROUP_SIZE:         return "CLFFT_INVALID_WORK_GROUP_SIZE";
    case CLFFT_INVALID_WORK_DIMENSION:          return "CLFFT_INVALID_WORK_DIMENSION";
    case CLFFT_INVALID_KERNEL_ARGS:             return "CLFFT_INVALID_KERNEL_ARGS";
    case CLFFT_INVALID_ARG_SIZE:                return "CLFFT_INVALID_ARG_SIZE";
    case CLFFT_INVALID_ARG_VALUE:               return "CLFFT_INVALID_ARG_VALUE";
    case CLFFT_INVALID_ARG_INDEX:               return "CLFFT_INVALID_ARG_INDEX";
    case CLFFT_INVALID_KERNEL:                  return "CLFFT_INVALID_KERNEL";
    case CLFFT_INVALID_KERNEL_DEFINITION:       return "CLFFT_INVALID_KERNEL_DEFINITION";
    case CLFFT_INVALID_KERNEL_NAME:             return "CLFFT_INVALID_KERNEL_NAME";
    case CLFFT_INVALID_PROGRAM_EXECUTABLE:      return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
    case CLFFT_INVALID_PROGRAM:                 return "CLFFT_INVALID_PROGRAM";
    case CLFFT_INVALID_BUILD_OPTIONS:           return "CLFFT_INVALID_BUILD_OPTIONS";
    case CLFFT_INVALID_BINARY:                  return "CLFFT_INVALID_BINARY";
    case CLFFT_INVALID_SAMPLER:                 return "CLFFT_INVALID_SAMPLER";
    case CLFFT_INVALID_IMAGE_SIZE:              return "CLFFT_INVALID_IMAGE_SIZE";
    case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CLFFT_INVALID_MEM_OBJECT:              return "CLFFT_INVALID_MEM_OBJECT";
    case CLFFT_INVALID_HOST_PTR:                return "CLFFT_INVALID_HOST_PTR";
    case CLFFT_INVALID_COMMAND_QUEUE:           return "CLFFT_INVALID_COMMAND_QUEUE";
    case CLFFT_INVALID_QUEUE_PROPERTIES:        return "CLFFT_INVALID_QUEUE_PROPERTIES";
    case CLFFT_INVALID_CONTEXT:                 return "CLFFT_INVALID_CONTEXT";
    case CLFFT_INVALID_DEVICE:                  return "CLFFT_INVALID_DEVICE";
    case CLFFT_INVALID_PLATFORM:                return "CLFFT_INVALID_PLATFORM";
    case CLFFT_INVALID_DEVICE_TYPE:             return "CLFFT_INVALID_DEVICE_TYPE";
    case CLFFT_INVALID_VALUE:                   return "CLFFT_INVALID_VALUE";
    case CLFFT_MAP_FAILURE:                     return "CLFFT_MAP_FAILURE";
    case CLFFT_BUILD_PROGRAM_FAILURE:           return "CLFFT_BUILD_PROGRAM_FAILURE";
    case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:      return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
    case CLFFT_IMAGE_FORMAT_MISMATCH:           return "CLFFT_IMAGE_FORMAT_MISMATCH";
    case CLFFT_MEM_COPY_OVERLAP:                return "CLFFT_MEM_COPY_OVERLAP";
    case CLFFT_PROFILING_INFO_NOT_AVAILABLE:    return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
    case CLFFT_OUT_OF_HOST_MEMORY:              return "CLFFT_OUT_OF_HOST_MEMORY";
    case CLFFT_OUT_OF_RESOURCES:                return "CLFFT_OUT_OF_RESOURCES";
    case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:   return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
    case CLFFT_COMPILER_NOT_AVAILABLE:          return "CLFFT_COMPILER_NOT_AVAILABLE";
    case CLFFT_DEVICE_NOT_AVAILABLE:            return "CLFFT_DEVICE_NOT_AVAILABLE";
    case CLFFT_DEVICE_NOT_FOUND:                return "CLFFT_DEVICE_NOT_FOUND";
    case CLFFT_SUCCESS:                         return "CLFFT_SUCCESS";
    case CLFFT_BUGCHECK:                        return "CLFFT_BUGCHECK";
    case CLFFT_NOTIMPLEMENTED:                  return "CLFFT_NOTIMPLEMENTED";
    case CLFFT_TRANSPOSED_NOTIMPLEMENTED:       return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
    case CLFFT_FILE_NOT_FOUND:                  return "CLFFT_FILE_NOT_FOUND";
    case CLFFT_FILE_CREATE_FAILURE:             return "CLFFT_FILE_CREATE_FAILURE";
    case CLFFT_VERSION_MISMATCH:                return "CLFFT_VERSION_MISMATCH";
    case CLFFT_INVALID_PLAN:                    return "CLFFT_INVALID_PLAN";
    case CLFFT_DEVICE_NO_DOUBLE:                return "CLFFT_DEVICE_NO_DOUBLE";
    case CLFFT_DEVICE_MISMATCH:                 return "CLFFT_DEVICE_MISMATCH";
    default:                                    return "Unknown error";
    }
  };

  return cformat("[clFFT error] %s", get());
}

#endif /* AFFT_DETAIL_GPU_CLFFT_ERROR_HPP */
