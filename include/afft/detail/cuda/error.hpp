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

#ifndef AFFT_DETAIL_GPU_CUDA_ERROR_HPP
#define AFFT_DETAIL_GPU_CUDA_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../exception.hpp"

namespace afft::detail::cuda
{
  /**
   * @brief Check if CUDA error is ok.
   * @param error CUDA error.
   * @return True if error is cudaSuccess, false otherwise.
   */
  [[nodiscard]] inline constexpr bool isOk(cudaError_t error)
  {
    return (error == cudaSuccess);
  }

  /**
   * @brief Check if CUDA error is valid.
   * @param error CUDA error.
   * @throw GpuBackendException if error is not valid.
   */
  inline void checkError(cudaError_t error)
  {
    if (!isOk(error))
    {
      throw GpuBackendException(cformatNothrow("%s - %s", cudaGetErrorName(error), cudaGetErrorString(error)));
    }
  }

  /**
   * @brief Check if CUDA error is ok.
   * @param result CUDA driver error.
   * @return True if result is CUDA_SUCCESS, false otherwise.
   */
  [[nodiscard]] inline constexpr bool isOk(CUresult result)
  {
    return (result == CUDA_SUCCESS);
  }

  /**
   * @brief Check if CUDA error is valid.
   * @param result CUDA driver error.
   * @throw GpuBackendException if result is not valid.
   */
  inline void checkError(CUresult result)
  {
    const char* errorName{};
    const char* errorStr{};

    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorStr);

    if (!isOk(result))
    {
      throw GpuBackendException{cformatNothrow("%s - %s", (errorName != nullptr) ? errorName : "unnamed error",
                                                          (errorStr != nullptr) ? errorName : "no description")};
    }
  }
} // namespace afft::detail::cuda

#endif /* AFFT_DETAIL_GPU_CUDA_ERROR_HPP */
