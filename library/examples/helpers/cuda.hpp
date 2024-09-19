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

#include "cuda.h"

namespace helpers::cuda
{
  /**
   * @brief Get the number of CUDA devices
   * @return Number of CUDA devices
   */
  inline int getDeviceCount()
  {
    return helpers_cuda_getDeviceCount();
  }

  /// @brief Deleter for CUDA streams
  struct StreamDeleter
  {
    void operator()(cudaStream_t stream) const
    {
      cudaStreamDestroy(stream);
    }
  };

  /// @brief Unique pointer for CUDA streams
  using Stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;

  /**
   * @brief Make a CUDA stream
   * @return CUDA stream
   */
  inline Stream makeStream(unsigned flags = cudaStreamDefault)
  {
    cudaStream_t stream{};

    CALL_CUDART(cudaStreamCreateWithFlags(&stream, flags));

    return Stream{stream};
  }
} // namespace helpers::cuda

#endif /* HELPERS_CUDA_HPP */
