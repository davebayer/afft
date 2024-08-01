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

#ifndef HELPERS_CUDA_HPP
#define HELPERS_CUDA_HPP

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <hip_runtime.h>

#include "cformat.hpp"

/**
 * @brief Macro for checking HIP errors. The call cannot contain _error variable.
 * @param call HIP function call
 */
#define HIP_CALL(call) \
  do { \
    const hipError_t _error = (call); \
    if (_error != hipSuccess) \
    { \
      throw cformatNothrow("HIP error (%s:%d) - %s", __FILE__, __LINE__, hipGetErrorString(_error)); \
    } \
  } while (0)

namespace helpers::hip
{
  /**
   * @brief Get the number of HIP devices
   * @return Number of HIP devices
   */
  inline int getDeviceCount()
  {
    int deviceCount{};

    HIP_CALL(hipGetDeviceCount(&deviceCount));

    return deviceCount;
  }

  /// @brief Deleter for HIP streams
  struct StreamDeleter
  {
    void operator()(hipStream_t stream) const
    {
      hipStreamDestroy(stream);
    }
  };

  /// @brief Unique pointer for HIP streams
  using Stream = std::unique_ptr<std::remove_pointer_t<hipStream_t>, StreamDeleter>;

  /**
   * @brief Make a HIP stream
   * @return HIP stream
   */
  inline Stream makeStream(unsigned flags = hipStreamDefault)
  {
    hipStream_t stream{};

    HIP_CALL(hipStreamCreateWithFlags(&stream, flags));

    return Stream{std::move(stream)};
  }
} // namespace helpers::hip

#endif /* HELPERS_HIP_HPP */
